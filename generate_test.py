import os

image_files = []
os.chdir(os.path.join("customdata", "images/valid/"))
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".jpg"):
        image_files.append("customdata/images/valid/" + filename)
os.chdir("..")
with open("valid-newcustom.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")