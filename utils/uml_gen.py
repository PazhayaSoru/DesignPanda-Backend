# import subprocess

# # Save UML code to file

# def uml_gen(uml_code : str):
    
#   with open("example.puml", "w") as f:
#       f.write(uml_code)

#   jar_path = "./plantuml-1.2025.2.jar"
#   java_path = "C:/Program Files/Java/jdk-23/bin/java.exe"
#   # Run PlantUML JAR to generate the diagram
#   # Make sure plantuml.jar is in the same directory or give full path
#   subprocess.run([java_path, "-jar", jar_path, "C:/Users/thede/OneDrive/Desktop/ete_backend/example.puml"], check=True)

#   print("✅ Diagram generated: example.png")

import subprocess
import os

def uml_gen(uml_code: str):
    # Define file paths
    puml_path = "./example.puml"
    output_dir = os.path.dirname(puml_path)
    
    # Save UML code to a .puml file
    with open(puml_path, "w") as f:
        f.write(uml_code)
    
    # Paths to PlantUML JAR and Java
    jar_path = "./plantuml-1.2025.2.jar"
    java_path = "C:/Program Files/Java/jdk-23/bin/java.exe"
    
    # Generate PNG using PlantUML
    subprocess.run([
        java_path,
        "-jar", jar_path,
        "-tpng",  # Specify PNG output
        puml_path
    ], check=True)

    print(f"Diagram saved to {output_dir} as PNG.")
