import os
import logging
import tempfile
import re
from flask import Flask, render_template, request, jsonify, send_file
import speech_recognition as sr
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from gtts import gTTS
import uuid
import subprocess

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

# List of supported languages for translation
SUPPORTED_LANGUAGES = {
    'af': 'Afrikaans',
    'ar': 'Arabic',
    'bn': 'Bengali',
    'de': 'German',
    'el': 'Greek',
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'hi': 'Hindi',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ml': 'Malayalam',
    'mr': 'Marathi',
    'nl': 'Dutch',
    'pa': 'Punjabi',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'sv': 'Swedish',
    'ta': 'Tamil',
    'te': 'Telugu',
    'tr': 'Turkish',
    'ur': 'Urdu',
    'zh-CN': 'Chinese (Simplified)'
}

# Map langdetect codes to deep_translator codes
LANG_CODE_MAP = {
    'zh-cn': 'zh-CN',  # Chinese simplified
    'zh-tw': 'zh-TW',  # Chinese traditional
    'hi': 'hi',        # Hindi (may be detected as another code)
    'ur': 'ur',        # Urdu
    'pa': 'pa',        # Punjabi
    'ta': 'ta',        # Tamil
    'te': 'te',        # Telugu
    'ml': 'ml',        # Malayalam
    'mr': 'mr',        # Marathi
    'bn': 'bn',        # Bengali
    'so': 'hi',        # Sometimes Hindi gets detected as Somali, map it to Hindi
}

# Languages that are commonly misdetected
COMMONLY_MISDETECTED = {
    'hi': ['so', 'ne', 'pa', 'mr'],  # Hindi can be detected as these
    'ur': ['ar', 'fa'],              # Urdu can be detected as Arabic or Persian
    'bn': ['as'],                    # Bengali can be detected as Assamese
}

# Function to improve basic grammar in transcribed text
def improve_grammar(text, language_code):
    """
    Applies basic grammar corrections to improve the transcribed text.
    
    Args:
        text (str): The text to improve
        language_code (str): Language code of the text
        
    Returns:
        str: Improved text
    """
    # Only apply corrections to English text
    if language_code != 'en':
        return text
        
    # Store original text
    original_text = text
    
    try:
        # Capitalize first letter of sentences
        text = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        # Fix common spacing issues
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,;:!?])([a-zA-Z])', r'\1 \2', text)  # Add space after punctuation
        
        # Fix common contractions
        contractions = {
            r'(\b)i(\b)': r'\1I\2',  # Capitalize standalone "i"
            r"(\b)im(\b)": r"\1I'm\2",
            r"(\b)cant(\b)": r"\1can't\2",
            r"(\b)dont(\b)": r"\1don't\2",
            r"(\b)didnt(\b)": r"\1didn't\2",
            r"(\b)wont(\b)": r"\1won't\2",
            r"(\b)cant(\b)": r"\1can't\2",
            r"(\b)havent(\b)": r"\1haven't\2",
            r"(\b)youre(\b)": r"\1you're\2",
            r"(\b)theyre(\b)": r"\1they're\2",
            r"(\b)its(\s+a|\s+the|\s+not)": r"\1it's\2",  # Only fix "its" to "it's" in specific contexts
        }
        
        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Log the correction if any change was made
        if text != original_text:
            logger.debug(f"Grammar improved: '{original_text}' -> '{text}'")
            
        return text
        
    except Exception as e:
        logger.warning(f"Error during grammar improvement: {e}")
        return original_text  # Return original text if anything goes wrong

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', languages=SUPPORTED_LANGUAGES)

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Transcribe audio uploaded from the client.
    Detect source language, transcribe to text.
    """
    temp_file = None
    wav_temp_file = None
    
    try:
        # Check if audio file was received
        if 'audio' not in request.files:
            logger.error("No audio file received")
            return jsonify({'error': 'No audio file received'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            logger.error("Empty audio file received")
            return jsonify({'error': 'Empty audio file received'}), 400

        logger.debug(f"Received audio file: {audio_file.filename}, Content Type: {audio_file.content_type}")
        
        # Save the audio file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
        audio_file.save(temp_file.name)
        temp_file.close()
        logger.debug(f"Saved audio to temporary file: {temp_file.name}")
        
        # Create a WAV file from the WebM as SpeechRecognition requires WAV format
        wav_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        wav_temp_file.close()
        
        # Use ffmpeg to convert WebM to WAV with enhanced quality settings
        logger.debug(f"Converting WebM to WAV with enhanced quality: {temp_file.name} -> {wav_temp_file.name}")
        
        # Run ffmpeg with more detailed output for debugging and improved quality settings
        try:
            # Enhanced conversion with noise reduction, normalization, and better sample rate
            ffmpeg_result = subprocess.run(
                [
                    'ffmpeg', '-y', 
                    '-i', temp_file.name, 
                    # Higher sample rate for better audio quality
                    '-ar', '44100',  
                    # Mono channel for speech recognition
                    '-ac', '1',  
                    # High quality PCM audio
                    '-c:a', 'pcm_s16le',
                    # Audio normalization to boost volume
                    '-af', 'loudnorm=I=-16:LRA=11:TP=-1.5',  
                    # Disable dynamic range compression which can distort speech
                    '-drc_scale', '0',
                    # Noise reduction filter for cleaner speech
                    '-af', 'highpass=f=200,lowpass=f=3000',
                    wav_temp_file.name
                ], 
                check=True, 
                capture_output=True,
                text=True
            )
            logger.debug(f"ffmpeg stdout: {ffmpeg_result.stdout}")
            logger.debug(f"ffmpeg stderr: {ffmpeg_result.stderr}")
            logger.debug("Enhanced audio conversion successful")
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg error: {e}")
            logger.error(f"ffmpeg stdout: {e.stdout}")
            logger.error(f"ffmpeg stderr: {e.stderr}")
            
            # If enhanced conversion fails, try simpler conversion as fallback
            logger.warning("Enhanced conversion failed, trying basic conversion as fallback")
            try:
                basic_result = subprocess.run(
                    ['ffmpeg', '-y', '-i', temp_file.name, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', wav_temp_file.name], 
                    check=True, 
                    capture_output=True,
                    text=True
                )
                logger.debug("Basic conversion successful as fallback")
            except subprocess.CalledProcessError as e2:
                logger.error(f"Basic ffmpeg fallback also failed: {e2}")
                return jsonify({'error': f'Audio conversion failed: {str(e2)}'}), 500
        
        # Initialize speech recognizer
        recognizer = sr.Recognizer()
        
        # Check if WAV file exists and has content
        if not os.path.exists(wav_temp_file.name):
            logger.error(f"WAV file does not exist: {wav_temp_file.name}")
            return jsonify({'error': 'WAV file not created'}), 500
            
        wav_size = os.path.getsize(wav_temp_file.name)
        logger.debug(f"WAV file size: {wav_size} bytes")
        
        if wav_size == 0:
            logger.error("WAV file is empty")
            return jsonify({'error': 'Empty audio file after conversion'}), 500
        
        # Use the WAV file with SpeechRecognition
        try:
            with sr.AudioFile(wav_temp_file.name) as source:
                logger.debug("Reading audio file")
                # Adjust for ambient noise to improve recognition accuracy
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Increase the energy threshold for better speech detection
                recognizer.energy_threshold = 300
                # Set dynamic energy threshold for varying volume levels
                recognizer.dynamic_energy_threshold = True
                # Increase pause threshold to allow for natural pauses in speech
                recognizer.pause_threshold = 0.8
                # Record the audio data with the adjusted settings
                audio_data = recognizer.record(source)
                logger.debug(f"Audio data recorded, frame count: {len(audio_data.frame_data)}")
            
            # Use Google's Speech Recognition to transcribe
            logger.debug("Sending to Google Speech Recognition")
            
            # Check if source_language is specified in the request
            source_language = request.form.get('source_language', 'auto')
            logger.debug(f"Source language from request: {source_language}")
            
            # Initialize variables to handle potential unbound variable issues
            detected_lang = None
            text = ""  # Initialize text to empty string as a safety measure
            
            # If a specific language is selected (not auto), use it for recognition
            if source_language != 'auto' and source_language in SUPPORTED_LANGUAGES:
                logger.debug(f"Using specified language for recognition: {source_language}")
                try:
                    # Try with a higher recognition accuracy setting
                    # We'll try multiple times with different settings if needed
                    success = False
                    try:
                        # First attempt - normal recognition
                        text = recognizer.recognize_google(audio_data, language=source_language)
                        success = True
                        logger.debug(f"Recognition succeeded with language {source_language}")
                    except sr.UnknownValueError:
                        # Second attempt - with show_all=True to get more detailed results
                        results = recognizer.recognize_google(audio_data, language=source_language, show_all=True)
                        if results and len(results.get('alternative', [])) > 0:
                            # Take the highest confidence result
                            text = results['alternative'][0]['transcript']
                            success = True
                            logger.debug(f"Recognition succeeded with alternative results in {source_language}")
                    
                    if success:
                        detected_lang = source_language  # Use the specified language
                    else:
                        # If both attempts fail, raise to trigger fallback
                        raise sr.UnknownValueError("No recognition results with sufficient confidence")
                        
                except sr.UnknownValueError:
                    # If recognition fails with specified language, fall back to auto-detection
                    logger.warning(f"Recognition failed with specified language {source_language}, falling back to auto-detection")
                    # Try auto-detection with show_all to get more detailed results
                    try:
                        results = recognizer.recognize_google(audio_data, show_all=True)
                        if results and len(results.get('alternative', [])) > 0:
                            text = results['alternative'][0]['transcript']
                            logger.debug(f"Auto-detection succeeded with alternative results")
                        else:
                            # Last resort - plain auto-detection
                            text = recognizer.recognize_google(audio_data)
                    except sr.UnknownValueError:
                        # If all attempts fail, raise again - this will be caught by the outer try/except
                        raise
                    # detected_lang will be set in the language detection block below
            else:
                # Use auto-detection with enhanced accuracy
                try:
                    # Try with show_all first to get the highest confidence result
                    results = recognizer.recognize_google(audio_data, show_all=True)
                    if results and len(results.get('alternative', [])) > 0:
                        text = results['alternative'][0]['transcript']
                        logger.debug(f"Auto-detection succeeded with alternative results")
                    else:
                        # Fall back to standard recognition
                        text = recognizer.recognize_google(audio_data)
                except sr.UnknownValueError:
                    # If all attempts fail, raise again - this will be caught by the outer try/except
                    raise
                    
            # Check if text is still empty after all attempts
            if not text:
                raise sr.UnknownValueError("Could not recognize any speech in the audio")
                
            logger.debug(f"Transcribed text: {text}")
            
            # If we're auto-detecting or if specified language recognition failed, detect from text
            if source_language == 'auto' or detected_lang is None:
                try:
                    detected_lang = detect(text)
                    logger.debug(f"Detected language (from text): {detected_lang}")
                    
                    # Handle commonly misdetected languages (like Hindi detected as Somali)
                    for actual_lang, possible_detections in COMMONLY_MISDETECTED.items():
                        if detected_lang in possible_detections:
                            logger.info(f"Detected language '{detected_lang}' is commonly confused with '{actual_lang}', using '{actual_lang}' instead")
                            detected_lang = actual_lang
                            break
                    
                    # Check if detected language is valid for translation
                    # If not in our supported languages, default to English
                    if detected_lang not in SUPPORTED_LANGUAGES and detected_lang not in LANG_CODE_MAP:
                        logger.warning(f"Detected language '{detected_lang}' not supported, defaulting to English")
                        detected_lang = 'en'
                except LangDetectException as e:
                    logger.warning(f"Language detection failed: {e}, defaulting to English")
                    detected_lang = 'en'
            
            # Map language code if needed
            if detected_lang in LANG_CODE_MAP:
                detected_lang = LANG_CODE_MAP[detected_lang]
                
            # Apply basic grammar improvements to the transcribed text
            original_text = text
            text = improve_grammar(text, detected_lang)
            if text != original_text:
                logger.info("Applied grammar improvements to transcribed text")
            
            return jsonify({
                'success': True,
                'text': text,
                'detected_language': detected_lang,
                'detected_language_name': SUPPORTED_LANGUAGES.get(detected_lang, 'Unknown')
            })
            
        except sr.UnknownValueError:
            logger.error("Google Speech Recognition could not understand audio")
            return jsonify({'error': 'Speech not recognized. Please speak clearly.'}), 400
            
        except sr.RequestError as e:
            logger.error(f"Google Speech Recognition service error: {e}")
            return jsonify({'error': f'Speech recognition service error: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Transcription error: {str(e)}'}), 500
        
    finally:
        # Clean up temporary files
        try:
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
                logger.debug(f"Deleted temp file: {temp_file.name}")
            
            if wav_temp_file and os.path.exists(wav_temp_file.name):
                os.unlink(wav_temp_file.name)
                logger.debug(f"Deleted WAV temp file: {wav_temp_file.name}")
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")

@app.route('/api/translate', methods=['POST'])
def translate_text():
    """
    Translate text from source language to target language.
    """
    try:
        data = request.json
        if not data or 'text' not in data or 'target_language' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        text = data['text']
        source_lang = data.get('source_language', 'auto')
        target_lang = data['target_language']
        
        logger.debug(f"Translating text from {source_lang} to {target_lang}: {text}")
        
        # Handle empty text
        if not text.strip():
            return jsonify({'error': 'Empty text cannot be translated'}), 400
        
        # Translate text
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated_text = translator.translate(text)
        
        logger.debug(f"Translated text: {translated_text}")
        
        if not translated_text:
            return jsonify({'error': 'Translation failed'}), 500
        
        return jsonify({
            'success': True,
            'translated_text': translated_text,
            'source_language': source_lang,
            'target_language': target_lang
        })
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return jsonify({'error': f'Translation error: {str(e)}'}), 500

@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    """
    Convert translated text to speech.
    """
    try:
        data = request.json
        if not data or 'text' not in data or 'language' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        text = data['text']
        language = data['language']
        
        logger.debug(f"Converting text to speech in {language}: {text}")
        
        # Handle empty text
        if not text.strip():
            return jsonify({'error': 'Empty text cannot be synthesized'}), 400
        
        # Generate unique filename
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        
        logger.debug(f"Saving speech to: {filepath}")
        
        # Convert text to speech
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(filepath)
        
        logger.debug(f"Speech file created: {filepath}")
        
        # Return audio file
        return send_file(
            filepath,
            mimetype='audio/mpeg',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Text-to-speech error: {e}")
        return jsonify({'error': f'Text-to-speech error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
