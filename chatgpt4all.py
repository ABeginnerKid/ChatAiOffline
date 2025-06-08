import os
import datetime
from gpt4all import GPT4All

# Ganti dengan nama file model Anda (pastikan file model sudah diunduh dan ada di direktori yang sesuai)
MODEL_PATH = "/home/arvan/CHAT/ggml-gpt4all-j-v1.3-groovy.bin"

# Buat direktori log jika belum ada
LOG_DIR = "chat_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Buat nama file log berdasarkan tanggal dan waktu saat ini
timestamp = datetime.datetime.now().strftime("%d%m%Y-%H%M")
log_filename = f"Chat{timestamp}.txt"
log_filepath = os.path.join(LOG_DIR, log_filename)

# Mulai sesi dengan model GPT4All
model = GPT4All(model_name=MODEL_PATH)

def log_chat(role, message):
    """Menuliskan pesan ke file log"""
    with open(log_filepath, "a", encoding="utf-8") as log_file:
        log_file.write(f"{role}: {message}\n")

print("GPT4All Long Conversation Mode")
print("Ketik 'exit' untuk keluar.\n")

with model.chat_session():
    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() == "exit":
                print("Mengakhiri sesi.")
                break

            log_chat("User", user_input)

            response = model.generate(user_input, max_tokens=200)  # Bisa diatur sesuai kebutuhan
            print(f"GPT: {response}")
            log_chat("GPT", response)

        except KeyboardInterrupt:
            print("\nSesi dihentikan oleh pengguna.")
            break
