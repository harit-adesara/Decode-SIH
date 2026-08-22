from .tools import government_scheme_rag


result = government_scheme_rag.invoke(
    {
        "query":"explain esanjeevani yojna"
    }
)

print(result)