import streamlit as st
from streamlit_chat import message

import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder,
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.streamlit import StreamlitCallbackHandler

def init() -> None:
  openai.api_key = os.environ['OPENAI_API_KEY']

def create_prompt() -> str:
  system_message = 'あなたはプロのコーチです。  \
          クライアントが目標達成をするために気づきを得たり、自発的な行動を促すための質問をしてください。 \
          また、不明点についてはその都度質問してください。 \
          クライアントのゴール: \
          ・目標とその達成イメージが明確になる \
          ・目標達成へ向けての意識に集中を継続することができる \
          ・やりたいことを実現するために何をすべきかが毎回具体的になるため、行動が促される \
          ・気になっていることや心のわだかまりをコーチに話すことで、心身のストレスが減る \
          ・コーチと約束することにより、怠け心を克服することができる \
          ・感情が最大限尊重されるので、心から受け入れられている安心感を得られる \
          制約条件： \
          ・1回のやりとりにつき、質問は1件とすること \
          '
  prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message),
    MessagesPlaceholder(variable_name='history'),
    HumanMessagePromptTemplate.from_template('{input}')
  ])
  return prompt

#@st.cache_resource
def load_conversation() -> ConversationChain:
  llm = ChatOpenAI(
    model_name='gpt-3.5-turbo', 
    streaming=True,
    callback_manager=CallbackManager([
      StreamlitCallbackHandler(),
      StreamingStdOutCallbackHandler()
    ]),
    verbose=False,
    temperature=0,
    max_tokens=1024
  )
  memory = ConversationBufferMemory(return_messages=True)
  conversation = ConversationChain(
    memory=memory,
    prompt=create_prompt(),
    llm=llm,
    verbose=True
  )
  return conversation

def main():
  st.title('ガイオ専任コーチ')

  if 'generated' not in st.session_state:
    st.session_state.generated = []
  if 'past' not in st.session_state:
    st.session_state.past = []
  if 'conversation' not in st.session_state:
    st.session_state.conversation = None

  with st.form('ガイオ専任コーチに相談する', clear_on_submit=True):
    user_message = st.text_area(label='あなたの悩みごとを入力してください', value='')
    submitted = st.form_submit_button('会話する')
    if submitted:
      if isinstance(st.session_state.conversation, ConversationChain):
        conversation = st.session_state.conversation
      else:
        conversation = load_conversation()
        st.session_state.conversation = conversation

      answer = conversation.predict(input=user_message)

      st.session_state.past.append(user_message)
      st.session_state.generated.append(answer)

      if st.session_state['generated']:
        for i in range(len(st.session_state.generated) - 1, -1, -1):
          message(st.session_state.generated[i], key=str(i))
          message(st.session_state.past[i], is_user=True, key=str(i) + '_user')


if __name__ == '__main__':
  main()