from call_llm import LLM

llm = LLM(mode='openai')

memory = ""
while True:
    print("-" * 100)
    prompt = input("You: ")
    print("-" * 100)

    if prompt == "xxx":
        memory = ""
        print("Memory wiped out.\n")
    else:
        memory += "\n Question: \n" + prompt
        answer = llm.call_llm(memory, big_model=False)
        print("-" * 100 + "\nLLM: " + answer + "\n" + "-" * 100)
        memory += "\n Answer: \n" + answer