from __future__ import division
import numpy as np


class IVABC:
    def __init__(self, n_particles, n_prots, mins, maxs, alpha, k,
                 max_iter=200, max_reps=50):
        self._n_particles = n_particles
        self._n_prots = n_prots
        self._mins = mins
        self._maxs = maxs
        self._alpha = alpha
        self._beta = 1.0 - alpha
        self._k = k
        self._max_iter = max_iter
        self._max_reps = max_reps

    def fit(self, X, y):
        self._initialize(X, y)
        t = 0
        repetitions = 0
        f_now = self._fitness_PBEST[self._index_GBEST]
        self.convergence_curve = np.zeros(self._max_iter)
        while t < self._max_iter and repetitions < self._max_reps:
            # print t
            self.convergence_curve[t] = f_now
            t += 1
            # atualizar PBESTs e GBEST
            self._send_worker_bees(X, y)
            # print "[{}, {}]".format(self._particles[24, 19, 20],
            #                         self._particles[24, 19, 21]
            #                         )
            self._send_onlooker_bees(X, y, t)
            # print "[{}, {}]".format(self._particles[24, 19, 20],
            #                         self._particles[24, 19, 21]
            #                         )
            self._send_scout_bees(X, y)
            # print "[{}, {}]".format(self._particles[24, 19, 20],
            #                         self._particles[24, 19, 21]
            #                         )

            f_new = self._fitness_PBEST[self._index_GBEST]
            if f_new == f_now:
                repetitions += 1
            else:
                repetitions = 0
            f_now = f_new
        return self

    def predict(self, X):
        predictions = _evaluate(X, None, self._PBEST[self._index_GBEST],
                                self._particle_labels[self._index_GBEST],
                                self._weights_PBEST[self._index_GBEST],
                                self._removed_PBEST[self._index_GBEST],
                                self._k, self._mins, self._maxs)
        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions != y)

    def _initialize(self, X, y):
        self._limits = np.zeros(self._n_particles)
        self._removed_prototypes = np.zeros((self._n_particles,
                                             np.sum(self._n_prots)), dtype=bool)
        self._particles = np.zeros((self._n_particles, np.sum(self._n_prots) +
                                    1, X.shape[1]))
        p = int(X.shape[1] / 2)
        self._particle_weights = np.ones((self._n_particles, np.sum(
            self._n_prots), p))
        self._velocities = np.random.uniform(-1, 1,
                          (self._n_particles, np.sum(
                          self._n_prots) + 1, 2 * p))
        self._particle_labels = np.zeros((self._n_particles, np.sum(
            self._n_prots)))
        self._fitness = np.zeros(self._n_particles)
        for particle in np.arange(self._n_particles):
            [self._particles[particle], self._particle_labels[particle],
             self._particle_weights[particle],
             self._removed_prototypes[particle]] = _create_particle(
                self._n_prots, X, y, self._mins, self._maxs, self._k)
            self._fitness[particle] = _calculate_fitness(
                X, y, self._particles[particle],
                self._particle_weights[particle],
                self._particle_labels[particle],
                self._removed_prototypes[particle],
                self._k, self._alpha, self._beta, self._mins, self._maxs)
        self._PBEST = np.copy(self._particles)
        self._fitness_PBEST = np.copy(self._fitness)
        self._weights_PBEST = np.copy(self._particle_weights)
        self._removed_PBEST = np.copy(self._removed_prototypes)
        self._index_GBEST = np.argmin(self._fitness)

    def _send_worker_bees(self, X, y):
        p = int(X.shape[1] / 2)
        len_prots = np.sum(self._n_prots)
        for particle in np.arange(self._n_particles):
            j = np.random.choice(p)  # selecionar uma dimensao aleatoria do problema
            prot = np.random.choice(
                np.append(np.delete(np.arange(len_prots),
                np.where(self._removed_prototypes[particle])) + 1, 0))  # selecionar um prototipo aleatorio do problema
            particle_k = np.random.choice(
                np.delete(np.arange(self._n_particles), particle))  # selecionar uma fonte aleatoria

            x_temp = _set_variable(self._particles, particle, particle_k,
                                   prot, j, self._mins, self._maxs)
            w = self._particle_weights[particle]
            if prot == 0:
                w = []
            [w, x_temp[1:], r_temp] = _calculate_prototype_weights(
                X, y, x_temp[1:], self._particle_labels[particle],
                x_temp[0, :p], self._k, self._n_prots, self._mins,
                self._maxs, self._removed_prototypes[particle], w
            )
            f = _calculate_fitness(X, y, x_temp, w,
                                   self._particle_labels[particle],
                                   r_temp, self._k, self._alpha, self._beta,
                                   self._mins, self._maxs)
            if f < self._fitness[particle]:
                self._fitness[particle] = f
                self._particles[particle] = x_temp
                self._particle_weights[particle] = w
                self._removed_prototypes[particle] = r_temp
                self._limits[particle] = 0
            else:
                self._limits[particle] += 1
            if f < self._fitness_PBEST[particle]:
                self._fitness_PBEST[particle] = f
                self._PBEST[particle] = x_temp
                self._weights_PBEST[particle] = w
                self._removed_prototypes[particle] = r_temp

        self._index_GBEST = np.argmin(self._fitness_PBEST)

    def _send_onlooker_bees(self, X, y, t):
        p = int(X.shape[1] / 2)
        fitness = 1.0 / (self._fitness + 1.0)
        probs = fitness / np.sum(fitness)

        wmin = 0.4
        wmax = 0.9

        w = wmin + (wmax - wmin) * (self._max_iter - t) / self._max_iter
        c1 = 0.5 + np.random.rand() / 2.0
        c2 = 0.5 + np.random.rand() / 2.0

        for index in np.arange(self._n_particles):
            particle = np.random.choice(self._n_particles, p=probs)
            r1 = np.random.rand()
            r2 = np.random.rand()

            self._velocities[particle] = np.clip(
                w * self._velocities[particle] + c1 * r1 * (
                self._PBEST[particle] - self._particles[particle]) + c2 * r2 * (
                self._PBEST[self._index_GBEST] - self._particles[particle]),
                -1, 1)

            used_vars_before = np.copy(self._particles[particle, 0, :p])
            x_temp = _fix_mins_maxs(
                _fix_used_variables(
                    self._particles[particle] + self._velocities[particle]),
                self._mins, self._maxs
            )  # calculo da nova posicao
            # ajuste da posicao para que os valores fiquem no intervalo [1;c], que sao os possiveis clusters
            we = self._particle_weights[particle]
            if np.any(np.around(used_vars_before) != np.around(x_temp[0, :p])):
                we = []
            [we, x_temp[1:], r_temp] = _calculate_prototype_weights(
                X, y, x_temp[1:], self._particle_labels[particle],
                x_temp[0, :p], self._k, self._n_prots, self._mins,
                self._maxs, self._removed_prototypes[particle], we
            )
            f = _calculate_fitness(X, y, x_temp, we,
                                   self._particle_labels[particle],
                                   r_temp, self._k, self._alpha, self._beta,
                                   self._mins, self._maxs)
            if f < self._fitness[particle]:
                self._fitness[particle] = f
                self._particles[particle] = x_temp
                self._particle_weights[particle] = we
                self._removed_prototypes[particle] = r_temp
                self._limits[particle] = 0
            else:
                self._limits[particle] += 1
            if f < self._fitness_PBEST[particle]:
                self._fitness_PBEST[particle] = f
                self._PBEST[particle] = x_temp
                self._weights_PBEST[particle] = we
                self._removed_prototypes[particle] = r_temp

        self._index_GBEST = np.argmin(self._fitness_PBEST)

    def _send_scout_bees(self, X, y):
        p = int(X.shape[1] / 2)
        for particle in np.arange(self._n_particles)[self._limits >= 10]:
            [self._particles[particle], self._particle_labels[particle],
             self._particle_weights[particle],
             self._removed_prototypes[particle]] = _create_particle(
                self._n_prots, X, y, self._mins, self._maxs, self._k)
            self._fitness[particle] = _calculate_fitness(X, y,
                                   self._particles[particle],
                                   self._particle_weights[particle],
                                   self._particle_labels[particle],
                                   self._removed_prototypes[particle],
                                   self._k, self._alpha, self._beta,
                                                         self._mins, self._maxs)

            self._velocities[particle] = np.random.uniform(-1, 1, (np.sum(
                self._n_prots) + 1, 2 * p))
            if self._fitness[particle] < self._fitness_PBEST[particle]:
                self._fitness_PBEST[particle] = self._fitness[particle]
                self._PBEST[particle] = np.copy(self._particles[particle])
                self._weights_PBEST[particle] = np.copy(self._particle_weights[
                                                            particle])
                self._removed_prototypes[particle] = self._removed_prototypes[
                    particle]
            self._limits[particle] = 0

        self._index_GBEST = np.argmin(self._fitness_PBEST)


def _create_particle(n_prots, X, y, mins, maxs, k):
    used_variables = np.random.rand(X.shape[1])
    p = int(X.shape[1] / 2)
    prototypes, prototype_labels = _select_prototypes(X, y, n_prots)

    particle = _fix_mins_maxs(
        _fix_used_variables(np.append([used_variables], prototypes, axis=0)),
        mins, maxs)
    [weights, particle[1:], removed_prototypes] = _calculate_prototype_weights(
        X, y, particle[1:], prototype_labels, used_variables[:p], k, n_prots,
        mins, maxs)
    return [particle, prototype_labels, weights, removed_prototypes]


def _calculate_fitness(X, y, particle, weights, particle_labels,
                       removed_prototypes, k, alpha, beta, mins, maxs):
    error, criterion, predictions = _evaluate(X, y, particle, particle_labels,
                                              weights, removed_prototypes, k,
                                              mins, maxs)
    return alpha * (error/100) + beta * criterion


def _select_prototypes_label(X, y, label, n_prots):
    chosen = np.random.choice(np.where(y == label)[0], n_prots)
    prototypes, prototype_labels = np.copy(X[chosen, :]), y[chosen]
    return prototypes, prototype_labels


def _select_prototypes(X, y, n_prots):
    labels = np.unique(y)
    prototypes = np.zeros((np.sum(n_prots), X.shape[1]))
    prototype_labels = np.zeros(np.sum(n_prots)).astype(int)
    end = 0
    for label in labels:
        init = end
        end += n_prots[label]
        p, l = _select_prototypes_label(X, y, label, n_prots[label])
        prototypes[init:end, :], prototype_labels[init:end] = p, l
    return prototypes, prototype_labels


def _fix_mins_maxs(data, mins, maxs):
    data[1:, ::2] = np.clip(data[1:, ::2], mins, maxs)
    data[1:, 1::2] = np.clip(data[1:, 1::2], mins, maxs)
    minima = data[1:, ::2]
    maxima = data[1:, 1::2]
    ind = minima > maxima
    if np.any(ind):
        temp = minima[ind]
        minima[ind] = maxima[ind]
        maxima[ind] = temp
        data[1:, ::2] = minima
        data[1:, 1::2] = maxima
    return data


def _fix_used_variables(particle):
    p = int(particle.shape[1] / 2)
    particle[0, :] = np.clip(particle[0, :], 0.0, 1.0)
    while np.all(np.around(particle[0, :p]) == 0):
        particle[0, :p] = np.random.rand(p)
    return particle


def _calculate_distances(prototypes, X, weights, used_variables,
                         removed_prototypes, mins, maxs):
    distances = np.ones((len(X), len(prototypes))) * np.inf
    for i, (prototype, w) in enumerate(zip(prototypes, weights)):
        if not removed_prototypes[i]:
            d_mins = np.sum(w * used_variables * (
                ((prototype[::2] - X[:, ::2]) / (maxs - mins)) ** 2.0), axis=1)
            d_maxs = np.sum(w * used_variables * (
                ((prototype[1::2] - X[:, 1::2]) / (maxs - mins)) ** 2.0),
                            axis=1)
            distances[:, i] = (d_mins + d_maxs) / (2.0 * np.sum(used_variables))
    return distances


def _calculate_prototype_weights(X, y, prototypes, prototype_labels,
                                 used_variables, k,
                                 n_prots, mins, maxs, removed_prototypes=[],
                                 weights=[]):
    labels = np.unique(y)
    n_labels = len(labels)
    used_var = np.around(used_variables)
    p = int(prototypes.shape[1] / 2)
    n = len(X)
    len_prots = len(prototypes)
    if len(removed_prototypes) == 0:
        removed_prototypes = np.zeros(len_prots, dtype=bool)
    if len(weights) == 0:
        weights = np.ones((len_prots, p))

    distances = _calculate_distances(prototypes, X, weights, used_var[:p],
                                     removed_prototypes, mins, maxs)
    n_removed = np.sum(removed_prototypes)
    k_nearest = np.argsort(distances, 1)[:, :min(k, len_prots - n_removed)]
    k_dists = distances[np.arange(n)[:, None], k_nearest]
    correct_labels = (prototype_labels[k_nearest] == y[:, None])
    correct_k_dists = correct_labels.astype(float) * k_dists

    number_correct = np.sum(correct_labels, 1)
    ne2 = np.sum(number_correct > 0)
    number_correct[number_correct == 0] = 1

    prototype_dists = np.array(
        [np.sum(correct_k_dists[k_nearest == prot]) for prot in
         np.arange(len_prots)])
    temp = ((2 * correct_labels - 1) * (k_nearest + 1)).ravel()
    occurrences = np.bincount(temp[temp >= 0], minlength=len_prots + 1)[1:]
    class_info = np.array([[np.sum(prototype_dists[prototype_labels == label]),
                          np.sum(np.logical_and((prototype_dists > 0),
                                                (prototype_labels == label))),
                            np.prod(prototype_dists[
    np.logical_and((prototype_dists > 0),
    (prototype_labels == label))] / occurrences[np.logical_and(
                                (prototype_dists > 0), (prototype_labels ==
                                                    label))])] for
                       label in np.arange(n_labels)])

    sum_classes = class_info[:, 0] / class_info[:, 1]
    sum_classes[np.isnan(sum_classes)] = 0.0
    if np.any(sum_classes == 0):
        class_weights = np.ones(n_labels)
        for label in labels:
            if sum_classes[label] == 0.0:
                class_members = prototype_labels == label
                [prototypes[class_members, :], prototype_labels[
                    class_members]] = _select_prototypes_label(X, y,
                                                               label,
                                                               n_prots[label])
                removed_prototypes[class_members] = False
                weights[class_members, :] = 1.0
    else:
        class_weights = (np.prod(sum_classes) ** (1.0 / n_labels)) / sum_classes
    prototype_weights = np.ones(len_prots)
    n_prototypes = class_info[:, 1]
    ind_prots = np.arange(len_prots)[np.logical_not(removed_prototypes)]
    prods_classes = class_info[:, 2]
    for prot in ind_prots:
        label = int(prototype_labels[prot])
        if sum_classes[label] > 0.0:
            if prototype_dists[prot] == 0:
                removed_prototypes[prot] = True
            else:
                prototype_weights[prot] = ((class_weights[label] *
                                            prods_classes[
                    label]) ** (1.0 / class_info[label, 1])) / (
                                        prototype_dists[prot] / occurrences[
                                            prot])
                members = np.where(np.logical_and((k_nearest == prot), (
                    correct_labels)))[0]
                mi = (((prototypes[prot, ::2] - X[members,
                                                ::2]) / (maxs - mins)) ** 2)
                ma = (((prototypes[prot, 1::2] - X[members,
                                                 1::2]) / (
                       maxs - mins)) ** 2)
                diff = (mi + ma) / (2 * np.sum(used_var))
                delta = np.sum(used_var * (diff / number_correct[members, None] /
                                              ne2), 0)
                found = np.any(np.logical_and((used_var == 1.0),
                                              (delta == 0.0)))
                if not found:
                    weights[prot, :] = np.array([((prototype_weights[prot] *
                                                   np.prod(
                        delta[delta > 0])) ** (1.0 / sum(
                        used_var))) / v if v > 0 else 1 for v in delta])
                else:
                    weights[prot, :] = np.array([prototype_weights[prot] ** (
                    1.0 / np.sum(used_var)) if v > 0 else 1 for v in
                                            used_var])
    return [weights, prototypes, removed_prototypes]


def _evaluate(X, y, particle, particle_labels, weights, removed_prototypes,
              k, mins, maxs):
    n = X.shape[0]
    p = int(X.shape[1] / 2)
    used_var = np.around(particle[0, :p])
    distances = _calculate_distances(particle[1:], X, weights, used_var,
                                     removed_prototypes, mins, maxs)
    labels = np.unique(y)
    if y is None:
        labels = np.unique(particle_labels)
    n_labels = len(labels)
    len_prots = len(removed_prototypes)
    n_removed = np.sum(removed_prototypes)
    k_nearest = np.argsort(distances, 1)[:, :min(k, len_prots - n_removed)]
    k_dists = distances[np.arange(n)[:, None], k_nearest]
    inverses = 1.0 / (k_dists + 1)
    members = np.array([particle_labels[k_nearest] == label for label in
                        np.arange(n_labels)])
    omegas = np.sum(members.astype(float) * inverses, 2).T
    winners = np.argmax(omegas, 1)
    zero_indices = np.where(k_dists == 0.0)[0]
    for ind in zero_indices:
        kd = k_dists[ind, :]
        if np.sum(kd == 0.0) == 1:
            winners[ind] = particle_labels[k_nearest[ind, 0]]
        else:
            winners[ind] = np.argmax(
                np.bincount(particle_labels[k_nearest[ind, kd == 0]].astype(
                    int),
                         minlength=n_labels))
    if y is not None:
        correct_labels = (particle_labels[k_nearest] == y[:, None])
        correct_k_dists = correct_labels.astype(float) * k_dists
        number_correct = np.sum(correct_labels, 1)
        ne2 = np.sum(number_correct > 0)
        number_correct[number_correct == 0] = 1
        criterion = np.sum(correct_k_dists / number_correct[:, None]) / ne2
        return np.mean(winners != y) * 100.0, criterion, winners
    else:
        return winners


def _set_variable(particles, particle, particle_k, prot, j, mins, maxs):
    fi = 2 * np.random.random_sample() - 1
    x_temp = np.copy(particles[particle])
    if prot == 0:
        var = particles[particle, prot, j] + fi * (
            particles[particle, prot, j] - particles[particle_k, prot, j])
        if var > 1:
            var = 1
        elif var < 0:
            var = 0
        x_temp[prot, j] = var
        x_temp = _fix_used_variables(x_temp)
    else:
        mi = particles[particle, prot, 2 * j] + fi * (
            particles[particle, prot, 2 * j] - particles[particle_k, prot,
                                                         2 * j])
        ma = particles[particle, prot, 2 * j + 1] + fi * (
            particles[particle, prot, 2 * j + 1] - particles[particle_k, prot,
                                                         2 * j + 1])
        if mi > ma:
            temp = ma
            ma = mi
            mi = temp
        mi = np.clip(mi, mins[j], maxs[j])
        ma = np.clip(ma, mins[j], maxs[j])
        x_temp[prot, 2 * j] = mi
        x_temp[prot, 2 * j + 1] = ma
    return x_temp