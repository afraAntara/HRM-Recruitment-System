import smtplib, ssl


def email_send(x):
    smtp_server = "smtp.gmail.com"
    port = 587  # For starttls
    #make this dynamic
    sender_email = "emp.site2021@gmail.com"
    password = input("Type your password and press enter: ")

    # Create a secure SSL context
    context = ssl.create_default_context()

    # Try to log in to server and send email
    try:
        server = smtplib.SMTP(smtp_server,port)
        server.ehlo() # Can be omitted
        server.starttls(context=context) # Secure the connection
        server.ehlo() # Can be omitted
        server.login(sender_email, password)
        server.sendmail(sender_email, 'emp.site2021@gmail.com', 'This is the message with OTP:' + str(x))

    except Exception as e:
        # Print any error messages to stdout
        print(e)
    finally:
        server.quit()