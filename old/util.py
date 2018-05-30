import smtplib

def sendMail(titulo, texto):
    gmail_user = "resultadosmeus@gmail.com"
    gmail_pwd = "SenhaDosResultados"
    FROM = 'resultadosmeus@gmail.com'
    TO = ['tmfilho@gmail.com'] #must be a list
    SUBJECT = titulo
    TEXT = texto
    # Prepare actual message
    message = """\From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        #server = smtplib.SMTP(SERVER)
        server = smtplib.SMTP("smtp.gmail.com", 587) #or port 465 doesn't seem to work!
        server.ehlo()
        server.starttls()
        server.login(gmail_user, gmail_pwd)
        server.sendmail(FROM, TO, message)
        #server.quit()
        server.close()
        print 'successfully sent the mail'
    except:
        print "failed to send mail"
        