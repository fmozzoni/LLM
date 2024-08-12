from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_parse import LlamaParse
import os
from langchain_community.chat_message_histories import ChatMessageHistory

import chromadb

BBDD = "./RAGS/gCCPublicas/"

os.environ["LLAMA_CLOUD_API_KEY"] = "llx-bIXJCGDIdrke54Kk3zd1B1ASCMqe9Pt0XuWGuMqQMEqkjPZm"

class cargadorLlamaIndex():
    def __init__(self):
        self.__rutaLectura = None
        self.__modelo = 'BAAI/bge-small-en-v1.5'
        self.__documento = None
        self.__cantidad = 0

    def setFuenteDatos(self, pRutaLectura):
        self.__rutaLectura = pRutaLectura

    def getFuenteDatos(self):
        return self.__rutaLectura

    def getCantidadVectores(self):
        print(self.__cantidad) 

    def leerFuenteDatos(self, pRutaLectura):
        self.setFuenteDatos(pRutaLectura)
        
        Settings.embed_model = HuggingFaceEmbedding(model_name= self.__modelo)

        print(self.getFuenteDatos())
        # Leer los documentos de las carpetas
        lParser = LlamaParse(
            api_key= "llx-bIXJCGDIdrke54Kk3zd1B1ASCMqe9Pt0XuWGuMqQMEqkjPZm",
            result_type= "markdown",
            verbose=True,
        )
        
        lExtencionLectura = { ".pdf" : lParser}
        self.__documento = SimpleDirectoryReader(input_dir=self.getFuenteDatos(), file_extractor= lExtencionLectura).load_data()
        
        # Configurar la base de datos de persistencia
        lDDBB = chromadb.PersistentClient(path=BBDD)

        # Configurar la colección
        lCollection = lDDBB.get_or_create_collection("Compras")

        # Configurar Chroma con la colección
        lVectorAlmacenamiento = ChromaVectorStore(chroma_collection=lCollection)
        lContextoAlmacenamiento = StorageContext.from_defaults(vector_store=lVectorAlmacenamiento)

        # Creación del Indice
        index = VectorStoreIndex.from_documents(documents=self.__documento, storage_context=lContextoAlmacenamiento, show_progress=True) 

        print(self.__documento[4].get_text())
        #for i in self.__documento:
        #   print (i)

        self.__cantidad = len(self.__documento) 
        return self.__cantidad


archivo=cargadorLlamaIndex()
print("Cantidad de archivos leidos: " + str(archivo.leerFuenteDatos("./DATOS/CCPublicas1")))
