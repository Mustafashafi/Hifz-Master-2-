import os
import time
import queue
import json
from fuzzywuzzy import fuzz
from vosk import Model, KaldiRecognizer
import sounddevice as sd
from IPython.display import display, HTML

# === Console Colors ===
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"
COLOR_GRAY = "\033[90m"

# === Colored Output in Jupyter ===
def show_colored_output(text, color="green"):
    color_map = {
        "green": "#00aa00",
        "red": "#cc0000",
        "blue": "#0044cc",
        "orange": "#cc6600",
        "purple": "#660066",
        "gray": "#888888"
    }
    display(HTML(f"<span style='color:{color_map.get(color, 'black')}; font-family:monospace'>{text}</span>"))

# === Load Quran from File ===
def load_quran(file_path):
    quran = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3:
                surah, ayah, text = parts
                surah = int(surah)
                ayah = int(ayah)
                if surah not in quran:
                    quran[surah] = {}
                quran[surah][ayah] = text
    return quran

# === Get Ayah by Surah and Ayah number ===
def get_next_ayah(quran, surah, ayah):
    return quran.get(surah, {}).get(ayah)

# === Highlight Differences in Analysis ===
def highlight_word_differences(expected_words, recited_words):
    colored = []
    for i in range(max(len(expected_words), len(recited_words))):
        ew = expected_words[i] if i < len(expected_words) else ""
        rw = recited_words[i] if i < len(recited_words) else ""
        score = fuzz.ratio(ew, rw)
        if not ew:
            colored.append(f"{COLOR_YELLOW}{rw}{COLOR_RESET}")
        elif not rw:
            colored.append(f"{COLOR_RED}[{ew}]{COLOR_RESET}")
        else:
            colored.append(f"{COLOR_GREEN if score >= 60 else COLOR_RED}{rw}{COLOR_RESET}")
    return ' '.join(colored)

def color_partial_partial(expected, partial, quran, surah_num, ayah_num, max_next_ayahs=10):
    expected_words = expected.split()
    
    # Append next Ayahs' words (up to max_next_ayahs)
    for i in range(1, max_next_ayahs + 1):
        next_ayah = get_next_ayah(quran, surah_num, ayah_num + i)
        if next_ayah:
            expected_words.extend(next_ayah.split())
        else:
            break

    recited_words = partial.strip().split()
    colored_words = []

    for i, rw in enumerate(recited_words):
        if i < len(expected_words):
            ew = expected_words[i]
            similarity_score = fuzz.ratio(ew, rw)
            color = COLOR_GREEN if similarity_score >= 60 else COLOR_RED
        else:
            color = COLOR_YELLOW  # Extra word, not expected
        colored_words.append(f"{color}{rw}{COLOR_RESET}")

    return ' '.join(colored_words)

def find_matching_previous_ayah(quran, surah, current_ayah, buffer_text, max_sequence=3, threshold=60):
    buffer_words = buffer_text.strip().split()
    best_match = None
    best_score = 0

    for length in range(max_sequence, 0, -1):
        for offset in range(1, current_ayah):
            start_ayah = current_ayah - offset
            end_ayah = start_ayah + length - 1
            if end_ayah >= current_ayah:
                continue

            sequence_words = []
            valid = True
            for ayah_num in range(start_ayah, end_ayah + 1):
                ayah_text = get_next_ayah(quran, surah, ayah_num)
                if not ayah_text:
                    valid = False
                    break
                sequence_words += ayah_text.strip().split()
            if not valid or len(sequence_words) > len(buffer_words):
                continue

            for i in range(len(buffer_words) - len(sequence_words) + 1):
                candidate = buffer_words[i:i + len(sequence_words)]

                if i > 0 and candidate == sequence_words:
                    continue

                correct = sum(fuzz.ratio(e, r) >= 60 for e, r in zip(sequence_words, candidate))
                accuracy = (correct / len(sequence_words)) * 100

                if accuracy >= threshold:
                    is_suffix_match = False
                    for prior in range(1, current_ayah):
                        prior_text = get_next_ayah(quran, surah, prior)
                        if prior_text and prior_text.strip().split()[-len(candidate):] == candidate:
                            is_suffix_match = True
                            break
                    if is_suffix_match and not (sequence_words == candidate):
                        continue

                    if accuracy > best_score or (accuracy == best_score and start_ayah < best_match['start']):
                        best_score = accuracy
                        best_match = {
                            "start": start_ayah,
                            "end": end_ayah,
                            "next": end_ayah + 1,
                            "remaining_buffer": ' '.join(buffer_words[i + len(sequence_words):]).strip()
                        }

    if best_match:
        print(f"{COLOR_YELLOW}↩️ Detected recitation of Ayahs {best_match['start']} to {best_match['end']} with accuracy {best_score:.2f}%. Jumping to Ayah {best_match['next']}!{COLOR_RESET}")
        return best_match['next'], best_match['remaining_buffer']

    return None, None

# === Find next Surah number if available ===
def get_next_surah_number(quran, current_surah):
    next_surahs = sorted([s for s in quran.keys() if s > current_surah])
    return next_surahs[0] if next_surahs else None

# === Get valid Surah number from user ===
def get_valid_surah_number():
    while True:
        try:
            surah_num = int(input("Enter Surah number to start from (1-114): "))
            if 1 <= surah_num <= 114:
                return surah_num
            print(f"{COLOR_RED}❌ Surah number must be between 1 and 114. Please try again.{COLOR_RESET}")
        except ValueError:
            print(f"{COLOR_RED}❌ Invalid input. Please enter a numeric Surah number.{COLOR_RESET}")

# === Main Recognition Loop ===
def recognize_from_microphone(quran):
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status, flush=True)
        q.put(bytes(indata))

    model_path = "E:/FYP/vosk-model-ar-0.22-linto-1.1.0"
    if not os.path.exists(model_path):
        raise Exception("Model not found at specified path!")

    model = Model(model_path)
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)

    # Get valid Surah number from user
    surah_num = get_valid_surah_number()
    ayah_num = 1
    
    print(f"\n{COLOR_BLUE}🎙️ Starting from Surah {surah_num}, Ayah {ayah_num}. Please begin recitation...{COLOR_RESET}")
    time.sleep(1)

    buffer_text = ""
    previous_partial = ""
    
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        print(f"{COLOR_GREEN}🔊 Listening...{COLOR_RESET}")
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                recited = result.get("text", "").strip()
                if not recited:
                    continue

                buffer_text += " " + recited
                buffer_text = buffer_text.strip()

                jump_to, remaining = find_matching_previous_ayah(quran, surah_num, ayah_num, buffer_text)
                if jump_to:
                    print(f"{COLOR_YELLOW}↩️ Jumping to Ayah {jump_to}.{COLOR_RESET}")
                    ayah_num = jump_to
                    buffer_text = remaining or ""
                    continue

                while True:
                    expected_ayah = get_next_ayah(quran, surah_num, ayah_num)
                    if not expected_ayah:
                        next_surah = get_next_surah_number(quran, surah_num)
                        if next_surah:
                            print(f"{COLOR_BLUE}📘 Surah {surah_num} completed. Moving to Surah {next_surah}, Ayah 1...{COLOR_RESET}")
                            surah_num = next_surah
                            ayah_num = 1
                            buffer_text = buffer_text.strip()
                            continue
                        else:
                            print(f"{COLOR_GREEN}✅ Quran recitation completed. No more Surahs available.{COLOR_RESET}")
                            return

                    expected_words = expected_ayah.strip().split()
                    recited_words = buffer_text.split()

                    if len(recited_words) < len(expected_words):
                        break

                    words_to_check = recited_words[:len(expected_words)]
                    correct_count = sum(fuzz.ratio(ew, rw) >= 60 for ew, rw in zip(expected_words, words_to_check))
                    accuracy = (correct_count / len(expected_words)) * 100

                    print(f"\n{COLOR_BLUE}📖 Ayah {ayah_num} Analysis:{COLOR_RESET}")
                    print(f"{COLOR_GRAY}Recited: {' '.join(words_to_check)}{COLOR_RESET}")
                    print(f"{COLOR_GRAY}Expected: {' '.join(expected_words)}{COLOR_RESET}")
                    print(f"Accuracy: {COLOR_GREEN if accuracy >= 60 else COLOR_RED}{accuracy:.2f}%{COLOR_RESET}")
                    print(f"🧠 Comparison: {highlight_word_differences(expected_words, words_to_check)}")

                    if accuracy >= 60:
                        print(f"{COLOR_GREEN}✅ Ayah {ayah_num} is correct! Moving to Ayah {ayah_num + 1}.{COLOR_RESET}")
                        ayah_num += 1
                        buffer_text = ' '.join(recited_words[len(expected_words):]).strip()
                        continue
                    else:
                        print(f"{COLOR_RED}⚠️ Mistakes detected. Please recite again.{COLOR_RESET}")
                        buffer_text = ''
                        break

            else:
                partial = json.loads(rec.PartialResult()).get("partial", "")
                if partial and partial != previous_partial:
                    expected_ayah = get_next_ayah(quran, surah_num, ayah_num)
                    if expected_ayah:
                        partial_colored = color_partial_partial(expected_ayah, partial, quran, surah_num, ayah_num)
                        print(f"\rPartial: {partial_colored}", end="", flush=True)
                    previous_partial = partial

# === Main ===
if __name__ == "__main__":
    quran_file_path = "E:/FYP/quran-simple.txt"
    quran = load_quran(quran_file_path)
    print(f"{COLOR_GREEN}✅ Quran loaded successfully.{COLOR_RESET}")
    recognize_from_microphone(quran)