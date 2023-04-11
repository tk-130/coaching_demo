# st_app.py
''' **ChatAPI APIの動作確認用コーチングデモモジュール**

Streamlit経由で簡易的なWebアプリとして動作する

使用方法：
    ・下記のように実行すると、ローカルでWebサーバが起動し、Webブラウザにリダイレクトされる
        >>> streamlit run  .\\st_app.py --server.port 5678

'''
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


class CoachingDemo:
    ''' コーチングデモアプリの実装クラス

    ブラウザで入力された文字列を受け取りChatGPT APIの呼び出しを行う

    Attributes
    ----------
    None
    '''

    def __init__(self) -> None:
        ''' コンストラクタ

        環境変数に設定されたOpenAIのAPIキーを設定する
        '''
        openai.api_key = os.environ['OPENAI_API_KEY']

    def create_prompt(self) -> str:
        ''' プロンプト作成処理

        ChatGPTの動作の前提を表すシステムプロンプト、過去のやりとり、
        ユーザ入力をAPIに渡せる形に整形し、返却する

        Returns
        ----------
        str
            APIに渡すプロンプト(文章)
        '''
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
                ・クライアントの目標が明らかになった場合、全ての回答に[目標:xx]というフォーマットでxxに目標を埋めて出力してください \
                ・クライアントの現状が明らかになった場合、全ての回答に[現状:yy]というフォーマットでyyに現状を埋めて出力してください \
                '
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message),
            MessagesPlaceholder(variable_name='history'),
            HumanMessagePromptTemplate.from_template('{input}')
        ])
        return prompt

    def load_conversation(self, **kwargs) -> ConversationChain:
        ''' 会話の実行処理

        指定された引数を元にChatGPT APIを呼び出し、会話のやりとりを返却する

        Parameters
        ----------
        kwargs : dict
            下記APIリファレンスの引数を持つ辞書
                https://platform.openai.com/docs/api-reference/chat/create

        Returns
        ----------
        ConversationChain
            会話のやりとりを記録したオブジェクト
        '''
        llm = ChatOpenAI(
            **kwargs,
            model_name='gpt-3.5-turbo',
            streaming=True,
            callback_manager=CallbackManager([
                StreamlitCallbackHandler(),
                StreamingStdOutCallbackHandler()
            ]),
            verbose=False
        )
        memory = ConversationBufferMemory(return_messages=True)
        conversation = ConversationChain(
            memory=memory,
            prompt=self.create_prompt(),
            llm=llm,
            verbose=True
        )
        return conversation

    def make_sidebar(self) -> dict:
        ''' サイドバーに関する処理

        ブラウザ上でサイドバーを描画し、ChatGPT APIに関するパラメータを指定するためのUIを描画する。
        また、指定された値を辞書型で返却する。

        Returns
        ----------
        dict
            下記APIリファレンスの引数を持つ辞書
                https://platform.openai.com/docs/api-reference/chat/create
        '''
        chat_args = dict()
        st.sidebar.subheader('ChatGPT APIパラメータ')
        chat_args['temperature'] = st.sidebar.slider(key='temperature',
                                                     label='文章のランダム性:(0-2)', min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        chat_args['top_p'] = st.sidebar.slider(key='top_p',
                                               label='文章の多様性:(0-1)', min_value=0.0, max_value=1.0, value=1.0, step=0.1)
        chat_args['stop'] = st.sidebar.text_input(key='stop',
                                                  label='終了条件', value=None)
        chat_args['max_tokens'] = st.sidebar.number_input(key='max_tokens',
                                                          label='最大トークン数(0-)', min_value=0, value=1024)
        chat_args['presence_penalty'] = st.sidebar.slider(key='pr_penalty',
                                                          label='同じ単語が繰り返し出現することの抑制:(-2-2)', min_value=-2.0,
                                                          max_value=2.0, value=0.0, step=0.1)
        chat_args['frequency_penalty'] = st.sidebar.slider(key='freq_penalty',
                                                           label='過去の予測で出現した回数に応じた単語の出現確率の引き下げ:(-2-2)',
                                                           min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
        return chat_args

    def main_proc(self) -> None:
        ''' メイン処理

        タイトル、フォーム等を描画し、ユーザ操作に関連した処理を呼び出す

        '''        
        st.title('ガイオ専任コーチ')

        if 'generated' not in st.session_state:
            st.session_state.generated = []
        if 'past' not in st.session_state:
            st.session_state.past = []
        if 'conversation' not in st.session_state:
            st.session_state.conversation = None
        if 'chat_args' not in st.session_state:
            st.session_state.chat_args = None

        st.session_state.chat_args = self.make_sidebar()
        with st.form('ガイオ専任コーチに相談する', clear_on_submit=True):
            user_message = st.text_area(label='あなたの悩みごとを入力してください', value='')
            submitted = st.form_submit_button('会話する')
            if submitted and user_message != '':
                
                if isinstance(st.session_state.conversation, ConversationChain):
                    conversation = st.session_state.conversation
                else:
                    chat_args = st.session_state.chat_args
                    conversation = self.load_conversation(**chat_args)
                    st.session_state.conversation = conversation

                answer = conversation.predict(input=user_message)

                st.session_state.past.append(user_message)
                st.session_state.generated.append(answer)

        if st.session_state['generated']:
            for i in range(len(st.session_state.generated) - 1, -1, -1):
                message(st.session_state.generated[i], key=str(i), avatar_style='bottts-neutral', seed='Aneka')
                message(st.session_state.past[i],
                        is_user=True, key=str(i) + '_user', avatar_style='pixel-art', seed='Aneka')


if __name__ == '__main__':
    CoachingDemo().main_proc()
