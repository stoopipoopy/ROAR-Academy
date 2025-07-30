with open("motto.txt", "w") as file:
    file.write("Fiat Lux!\n")

with open("motto.txt", "r") as file:
    content = file.read()
    print(content)

with open("motto.txt", "a") as file:
    file.write("Let there be light!\n")