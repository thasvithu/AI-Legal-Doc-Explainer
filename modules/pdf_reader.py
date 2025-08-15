from langchain.document_loaders import PyMuPDFLoader


loader = PyMuPDFLoader("../data/Sample_Contract_1.pdf")
docs = loader.load()
docs