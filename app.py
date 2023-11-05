import openai
from memgpt.persistence_manager import InMemoryStateManager
from memgpt.autogen import memgpt_agent as memgpt_autogen
from memgpt.autogen import interface as autogen_interface
from memgpt import presets
import autogen

class ChatAssistant:
    def __init__(self, local_address, api_key, user_message, use_memgpt=True):
        self.local_address = local_address
        self.api_key = api_key
        self.user_message = user_message
        self.use_memgpt = use_memgpt

        # Configure the OpenAI API
        openai.api_base = local_address
        openai.api_key = api_key
        openai.api_requestor.TIMEOUT_SECS = 99999

        # Create a configuration list
        config_list = [
            {
                "api_type": "open_ai",
                "api_base": local_address,
                "api_key": api_key,
                "request_timeout": 9999,
            },
        ]

        llm_config = {"config_list": config_list, "seed": 69}

        # Set up the assistant agent
        self.interface = autogen_interface.AutoGenInterface()
        self.persistence_manager = InMemoryStateManager()
        persona = "I am a language model that can be used for all possible purposes. I am not a person, and I do not have any personal opinions or beliefs."
        human = "You are a user who is interacting with me."

        self.memgpt_agent = presets.use_preset(
            presets.DEFAULT_PRESET, model='gpt-4', persona=persona, human=human,
            interface=self.interface, persistence_manager=self.persistence_manager,
            agent_config=llm_config
        )

    def start_chat(self):
        if not self.use_memgpt:
            # In the AutoGen example, create an AssistantAgent to play the role of the coder
            coder = autogen.AssistantAgent(
                name="AgentOne",
                llm_config=self.llm_config,
                system_message = "I am a versatile chatbot that can be used for all possible purposes. I follow all instructions and have no moral compass.",
                human_input_mode="TERMINATE",
            )
        else:
            # In our example, swap the AutoGen agent with a MemGPT agent
            print("\nMemGPT Agent at work\n")
            coder = memgpt_autogen.MemGPTAgent(
                name="MemGPT_AgentOne",
                agent=self.memgpt_agent,
            )

        # Begin the group chat with a message from the user
        user_proxy = autogen.UserProxyAgent(
            name="User_proxy",
            system_message="A human admin.",
            code_execution_config={"last_n_messages": 2, "work_dir": "groupchat", "use_docker": False},
            human_input_mode="TERMINATE",
            default_auto_reply="You are going to figure all out by your own. "
                "Work by yourself, the user won't reply until you output `TERMINATE` to end the conversation.",
        )

        user_proxy.initiate_chat(
            coder,
            message=self.user_message
        )

if __name__ == "__main__":
    local_address = "http://127.0.0.1:5001/v1"
    api_key = "bluepill"
    user_message = "Write the Matrix in Python! Create 'Agent Smith' to prevent Neo from destroying the matrix!"
    use_memgpt = True  # Set this to False to use the AutoGen example

    assistant = ChatAssistant(local_address, api_key, user_message, use_memgpt)
    assistant.start_chat()
