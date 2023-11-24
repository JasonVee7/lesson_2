import importlib

# Check if mediapipe is installed
try:
    import mediapipe as mp
    mediapipe_installed = True
except ImportError:
    mediapipe_installed = False

if not mediapipe_installed:
    print("mediapipe is not installed.")

    # Ask the user if they want to install mediapipe
    user_input = input("Do you want to install mediapipe? (y/n): ")

    if user_input.lower() == 'y':
        # Install mediapipe using pip
        try:
            import subprocess
            subprocess.run(["pip", "install", "mediapipe"])
            importlib.reload(mp)  # Reload the module to use the installed mediapipe
            print("mediapipe has been installed successfully.")
        except Exception as e:
            print("Error installing mediapipe:", e)
    else:
        print("mediapipe installation skipped.")

# Continue with your code after checking for mediapipe
if mediapipe_installed:
    print('test')
