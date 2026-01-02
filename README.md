# 🤖 RAG Tabanlı Çoklu-Model Doküman Asistanı

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Gemini](https://img.shields.io/badge/Google-Gemini%201.5-yellow)

Bu proje, **"Üretken Yapay Zeka ile Chatbot Geliştirme"** dersi kapsamında hazırlanmış, kullanıcıların yüklenen PDF dokümanları ile doğal dilde sohbet etmesini sağlayan gelişmiş bir yapay zeka asistanıdır.

Proje, **RAG (Retrieval-Augmented Generation)** mimarisini kullanarak modelin halüsinasyon görmesini engeller ve sadece dokümandaki verilere dayalı cevaplar üretir. Ayrıca kullanıcıya **Google Gemini** ve **OpenAI GPT** modelleri arasında seçim yapma imkanı sunar.

## 🚀 Özellikler

* **📄 Doküman Analizi:** PDF dosyalarını (Örn: One Piece Wiki, Makaleler) okur, parçalar ve vektörize eder.
* **🧠 Çoklu Model Desteği:** Kullanıcı, arayüz üzerinden **Google Gemini 1.5 Flash** (Ücretsiz/Hızlı) veya **OpenAI GPT-3.5** modellerinden birini seçebilir.
* **🛡️ Halüsinasyon Önleme:** `temperature=0` ayarı ve özel sistem talimatları (System Prompt) ile modelin uydurma yapması engellenmiştir.
* **⚡ Hızlı Erişim:** ChromaDB vektör veritabanı ve önbellekleme (Caching) sayesinde sorulara milisaniyeler içinde yanıt verir.
* **💻 Kullanıcı Dostu Arayüz:** Streamlit ile geliştirilmiş modern ve sade bir web arayüzü.

## 🎥 Proje Tanıtım Videosu

Projenin nasıl çalıştığını, model geçişlerini ve soru-cevap performansını aşağıdaki videodan izleyebilirsiniz:

[👉 **TANITIM VİDEOSUNU İZLEMEK İÇİN TIKLAYIN**](BURAYA_YOUTUBE_LINKINI_YAPISTIR)

---

## 🛠️ Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin.

### 1. Projeyi Klonlayın
```bash
git clone [https://github.com/saresarey/chatbot-project.git](https://github.com/saresarey/chatbot-project.git)
cd chatbot-project