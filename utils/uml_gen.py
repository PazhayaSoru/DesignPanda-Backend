import subprocess

# Save UML code to file

def uml_gen(uml_code : str):
    
  with open("example.puml", "w") as f:
      f.write(uml_code)

  jar_path = "./plantuml-1.2025.2.jar"
  java_path = "C:/Program Files/Java/jdk-21/bin/java.exe"
  # Run PlantUML JAR to generate the diagram
  # Make sure plantuml.jar is in the same directory or give full path
  subprocess.run([java_path, "-jar", jar_path, "C:/Users/thede/OneDrive/Desktop/ete_backend/example.puml"], check=True)

  print("✅ Diagram generated: example.png")