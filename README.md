# 🏴‍☠️ Going-Chaty: One Piece RAG Asistanı

**Geliştirici:** R. Sare Yılmaz  
**Durum:** 🚀 Aktif Geliştirme

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python) 
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat&logo=streamlit)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange?style=flat)
![RAG](https://img.shields.io/badge/Hybrid-RAG-green?style=flat)

## 🌊 Proje Hakkında

**Going-Chaty**, standart bir doküman asistanından çok daha fazlasıdır. O, yüklediğiniz PDF dokümanlarına (örneğin One Piece lore'u) hakim olan, **LangGraph** mimarisiyle düşünen ve **One Piece tayfasının ruhunu taşıyan** akıllı bir asistandır.

Bu proje, **Hybrid RAG (Retrieval-Augmented Generation)** yapısını kullanır:
* **Hafıza (Embeddings):** Google API hız sınırlarına takılmamak ve verimlilik için yerel işlemci gücü (**Local CPU - HuggingFace**) kullanılır.
* **Zeka (LLM):** Yaratıcı ve doğru cevaplar için Google **Gemini 3.0 Flash** modelinin gücünden faydalanır.

### 🎥 Proje Demosu

https://github.com/user-attachments/assets/c2a88028-8dc9-4213-afbe-5f45fca4afa1

---

## 🚀 Öne Çıkan Özellikler

### 🧠 Akıllı Hafıza & Oturum Yönetimi
* **Sohbet Geçmişi:** Konuştuğunuz her şey JSON formatında kaydedilir. Uygulamayı kapatsanız bile sohbetleriniz kaybolmaz.
* **Oturum Yönetimi:** Yan menüden eski sohbetlerinize dönebilir, onları yeniden adlandırabilir veya silebilirsiniz.
* **Context Awareness:** "Luffy kimdir?" dedikten sonra "Peki gemisi ne?" diye sorarsanız, kimden bahsettiğinizi anlar.

### 🏴‍☠️ One Piece "Ruhu" (Persona)
* **Dinamik Tepkiler:** Bot sadece cevap vermez; sizi "Kaptan" diye selamlar, One Piece tarzı gülüşler (Shishishi, Yohoho) kullanır.
* **Canlı Yükleme Ekranı:** Cevap beklerken sıkıcı bir dönen çark yerine *"Sanji yemek yapıyor...", "Zoro yine kayboldu..."* gibi rastgele durum mesajları görürsünüz.
* **Rastgele Konum:** Her açılışta tayfa farklı bir adadadır (Egghead, Wano, Elbaf vb.).

### 🎨 Özel Arayüz (UI)
* **WhatsApp Tarzı Görünüm:** Standart Streamlit ikonları kaldırıldı. Mesajlar sağa/sola yaslı şık baloncuklar içinde gösterilir.
* **İnteraktif Menü:** Sohbetleri yönetmek için modern "Üç Nokta" menüsü ve açılır pencereler (Dialog) kullanılır.

### ⚙️ İleri Teknoloji (LangGraph)
* Eski usul "Zincir (Chain)" yapısı yerine, kararları ve akışı yöneten **LangGraph (Node & Edge)** yapısı kullanılmıştır. Bu sayede botun düşünme süreci modülerdir (`Retrieve` -> `Generate`).

---

## 🛠️ Teknik Altyapı

| Bileşen | Teknoloji | Açıklama |
| :--- | :--- | :--- |
| **Dil (Language)** | Python 3.12 | Ana geliştirme dili. |
| **Arayüz (UI)** | Streamlit | Chat arayüzü ve oturum yönetimi. |
| **Orkestrasyon** | **LangGraph** | Durum yönetimi (State Management) ve akış kontrolü. |
| **LLM** | Gemini 3.0 Flash | Cevap üretimi (Generative AI). |
| **Embeddings** | all-MiniLM-L6-v2 | **Yerel & Ücretsiz.** PDF'i vektöre çeviren model. |
| **Veritabanı** | ChromaDB | Vektör verilerinin tutulduğu yerel veritabanı. |

---

## ⚙️ Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için:

### 1. Projeyi Klonlayın
```bash
git clone [https://github.com/saresarey/chatbot-project.git](https://github.com/saresarey/chatbot-project.git)
cd chatbot-project
