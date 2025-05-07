import json
import os
import tempfile

_temp_credentials_file = None

def setup_google_credentials():
    """
    Checks for GOOGLE_APPLICATION_CREDENTIALS content, saves it to a temp file,
    and updates the environment variable to point to the file path.
    """
    global _temp_credentials_file
    credentials_content = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_CONTENT') # Use a different name initially


    try:
        json.loads(credentials_content) # Raises json.JSONDecodeError if invalid

        with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as temp_file:
            temp_file.write(credentials_content)
            _temp_credentials_file = temp_file.name # Store the path globally


        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = _temp_credentials_file


    except Exception as e:
        raise RuntimeError("Failed to configure Google credentials from environment content.") from e