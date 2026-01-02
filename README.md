# 🤖 Çoklu Model Doküman Asistanı (Hybrid RAG Chatbot)

**Ders:** MTH 409 - Üretken Yapay Zeka Yardımı ile Chatbot Geliştirme Temelleri  
**Öğrenci:** Rahmet Sare Yılmaz  
**Tarih:** 02.01.2026

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red) ![LangChain](https://img.shields.io/badge/LangChain-Hybrid-green)

## 📄 Proje Hakkında
Bu proje, kullanıcıların yüklediği PDF dokümanları üzerinden doğal dilde soru-cevap yapabilen akıllı bir asistandır.

Proje, **RAG (Retrieval-Augmented Generation)** mimarisini kullanır. Ancak standart RAG uygulamalarından farklı olarak **Hibrit (Hybrid)** bir yapıya sahiptir:
1.  **Hafıza (Embedding):** Maliyet ve API hız sınırlarını (Rate Limit) aşmak için yerel işlemci gücü (**HuggingFace - Local CPU**) kullanılır.
2.  **Zeka (LLM):** Cevap üretmek için Google'ın **Gemini-Pro** modeli kullanılır.

Bu sayede proje hem **ücretsiz** hem de **yüksek performanslı** çalışır.

## 🚀 Özellikler
* **PDF Analizi:** Kullanıcı kendi PDF dosyasını yükleyebilir.
* **Vektör Veritabanı:** Dokümanlar parçalanarak ChromaDB üzerinde vektörel olarak saklanır.
* **Kaynak Gösterimi:** Bot, verdiği cevabı dokümanın hangi sayfasından aldığını gösterir.
* **Sohbet Yeteneği:** Sadece teknik sorulara değil, selamlaşma ve vedalaşma gibi sosyal etkileşimlere de cevap verir.
* **Çoklu Model Desteği:** Altyapı hem Google Gemini hem de OpenAI GPT modellerini destekler.

## 🛠️ Kullanılan Teknolojiler
* **Python 3.12.8**
* **Arayüz:** Streamlit
* **Orkestrasyon:** LangChain
* **LLM (Model):** Google Gemini Pro (`gemini-pro`)
* **Embeddings (Vektör):** HuggingFace (`all-MiniLM-L6-v2`) - *Yerel ve Ücretsiz*
* **Veritabanı:** ChromaDB
* **Güvenlik:** Python-dotenv

## ⚙️ Kurulum

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

### 1. Projeyi Klonlayın
```bash
git clone [https://github.com/kullaniciadi/proje-ismi.git](https://github.com/kullaniciadi/proje-ismi.git)
cd proje-ismi