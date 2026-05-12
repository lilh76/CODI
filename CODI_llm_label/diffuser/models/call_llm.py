import random
import openai
import time

class LLM:
    def __init__(self, mode='openai') -> None:
        if mode == 'openai':
            api_key_list = [
                '', # fill the key
            ]
            self.agent_big = gpt_agent(random.choice(api_key_list), api_key_list, model_name='gpt-4o-2024-08-06')
            self.agent_small = gpt_agent(random.choice(api_key_list), api_key_list, model_name='gpt-4o-mini')
            self.call_llm = self.call_llm_openai
        else:
            assert 0
    
    def call_llm_openai(self, prompt, big_model=False, temperature=0.0):
        if big_model:
            return self.agent_big.ask(prompt)
        else:
            return self.agent_small.ask(prompt)

class gpt_agent():

    def __init__(self, api_key:str, api_key_list, model_name="gpt-3.5-turbo"):
        # import openai
        openai.api_base = "https://api.openai.com/v1"
        openai.api_key = api_key # a key string
        self.api_key = api_key
        self.ask_call_cnt = 0
        self.ask_call_cnt_sup = 3
        self.model_name = model_name
        self.api_key_list = api_key_list

    def ask(self, question, temperature=0.0, stop=None) -> str:
        res = "No answer!"
        self.ask_call_cnt = self.ask_call_cnt + 1
        if self.ask_call_cnt > self.ask_call_cnt_sup:
            print("======> Achieve call count limit, Return!")
            self._random_key()
            self.ask_call_cnt = 0
            return res

        messages = [{"role": "user", "content": question}]
        try:
            rsp = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                # temperature=temperature,
                stop=stop
            )
            res = rsp.get("choices")[0]["message"]["content"]
            self.ask_call_cnt = 0
        except openai.error.AuthenticationError as e:
            self._random_key()
            print("======> openai.error.AuthenticationError", e)
        except openai.error.RateLimitError as e:
            print(f"======> {self.api_key} <===== \nAchieve ChatGPT rate limit, sleep!", e)
            self._random_key()
            time.sleep(10)
            return self.ask(question)
        except openai.error.ServiceUnavailableError:
            print('======> Service unavailable error: will retry after 10 seconds')
            self._random_key()
            time.sleep(10)
            return self.ask(question)
        except Exception as e:
            print("======> Exception occurs!", e)
            self._random_key()
            if "HTTPSConnectionPool" in str(e.error):
                time.sleep(60)
        return res

    def _random_key(self) -> None:
        self.api_key = random.choice(self.api_key_list)
        openai.api_key = self.api_key
