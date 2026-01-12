import chromadb

client = chromadb.PersistentClient(path="./vectorstore")
                                   
collection = client.get_or_create_collection(name="programlama")
print("Veritabanı ve koleksiyon başarıyla hazırlandı")                                  