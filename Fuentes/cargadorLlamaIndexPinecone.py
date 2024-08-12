from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
from llama_parse import LlamaParse
from pinecone import Pinecone
from pinecone import ServerlessSpec
from llama_index.core.node_parser import SimpleNodeParser

import os
import traceback
import pinecone

BBDD = "./RAGS/gCCPublicas/"

os.environ["LLAMA_CLOUD_API_KEY"] = "llx-bIXJCGDIdrke54Kk3zd1B1ASCMqe9Pt0XuWGuMqQMEqkjPZm"
PINECONE_KEY = Pinecone(api_key = "1e2f7eea-a8e6-4f60-a50b-2596b3f595f9")

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

        try:
            Paso = 0
            self.setFuenteDatos(pRutaLectura)
            
            # Leer los documentos de las carpetas
            Paso+=1
            lParser = LlamaParse(
                api_key= "llx-bIXJCGDIdrke54Kk3zd1B1ASCMqe9Pt0XuWGuMqQMEqkjPZm",
                result_type= "markdown",
                verbose=True,
            )

            Paso+=1
            lExtencionLectura = { ".pdf" : lParser}
            self.__documento = SimpleDirectoryReader(input_dir=self.getFuenteDatos(), file_extractor= lExtencionLectura).load_data()
            
            # Configurar el indice en Pinecone - La dimension debe coicider con la especificacion del embeddings
            Paso+=1

            lIndice = 'compras'
            if lIndice not in PINECONE_KEY.list_indexes().names():
                PINECONE_KEY.create_index (
                    name=lIndice,
                    dimension=768,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                
            Paso+=1
            lPinecone_index = PINECONE_KEY.Index(lIndice)

            Paso+=1
            lVectorAlmacenamiento = PineconeVectorStore(pinecone_index=lPinecone_index)
            lContextoAlmacenamiento = StorageContext.from_defaults(vector_store=lVectorAlmacenamiento)

            Settings.embed_model = HuggingFaceEmbedding(model_name=self.__modelo)            

            Paso+=1
            index = VectorStoreIndex.from_documents(documents=self.__documento, storage_context=lContextoAlmacenamiento, show_progress=True)

            #for i in self.__documento:
            #   print (i)

            Paso+=1
            self.__cantidad = len(self.__documento) 
            return self.__cantidad
        
        except Exception:
            traceback.print_exc()   


archivo=cargadorLlamaIndex()
print("Cantidad de archivos leidos: " + str(archivo.leerFuenteDatos("./DATOS/CCPublicas1")))
