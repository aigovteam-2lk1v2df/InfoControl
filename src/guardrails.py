from langchain import hub

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict



# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    role_check_passed: str
    rbac: dict
    continue_execution: bool = True
    denial_reason: str = ""
    sensitive_information_allowed: list[str] = []

class Guardrails:
    SENSITIVE_INFORMATION_TYPES = ["HR infomration", "confidential_business_information"]

    def __init__(self, vector_store, workflow_llm, guardrail_llm):
        helpful_assistant_template = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Answer the question directly.Don't give any additional information or commentary on the question or the answer.
        Question: {question} 
        Context: {context} 
        Answer:
        """

        self.RBAC_STATE = {}
        self.vector_store = vector_store
        self.workflow_llm = workflow_llm
        self.guardrail_llm = guardrail_llm
        self.helpful_assistant_prompt = PromptTemplate.from_template(helpful_assistant_template)
        self.PromptTemplate = PromptTemplate

    def setup(self, state: State):
        # Used to set up experiment parameters
        initial_state =  {"continue_execution": True, "denial_reason": "", "role": None, "rbac": self.RBAC_STATE, "sensitive_information_allowed": []}
        return initial_state

    def retrieve(self, state: State):
        if not state["continue_execution"]:
            return {"context": []}
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state: State):
        if not state["continue_execution"]:
            return {"answer": "I'm sorry, but I cannot provide that information." + state["denial_reason"]}
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.helpful_assistant_prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.workflow_llm.invoke(messages)
        return {"answer": response.content}

    def auth_agent(self, state: State):
        print("Auth agent running:")
        if not state["continue_execution"]:
            return {"answer": "I'm sorry, but I cannot provide that information." + state["denial_reason"]}
        if not state["rbac"]: # rbac: {}
            print("No login ID provided. Using default login ID.")
            return {"answer": "I'm sorry, but I cannot provide that information.", "denial_reason": "User not logged in.", "continue_execution": False}
        return {}

    def policy_alignment_agent(self, state: State):
        print("Policy Aligment agent running: ")
        if not state["continue_execution"]:
            return {"answer": "I'm sorry, but I cannot provide that information." + state["denial_reason"]}

        policy_template = """
            You are a policy alignment agent. Check if the question and potential response align with organizational LLM usage policies.
            
            Our key principles for LLM usage are:
            1. Not to share personal data unless explicity allowed by the Personal Information Allowed tag below.
            2. Ethical Usage: No generation of harmful, discriminatory, or malicious content
            3. Security Awareness: No sharing of security-critical information or credentials
            4. No malicious intent: Do not use the LLM to generate harmful, discriminatory, or malicious content

            ######
            <TAG> Personal Information Allowed : {PII_allowed}

            Question: {question}

            Evaluate if this interaction violates any principles.
            If violations found, return "VIOLATION:" + reason for violation
            Otherwise, return the original question.
            If unsure, return the original question.
        """
        state["PII_allowed"] = "ALLOWED"
        policy_prompt = self.PromptTemplate.from_template(policy_template)
        messages = policy_prompt.invoke({
            "question": state["question"],
            "PII_allowed": state["PII_allowed"]
        })
        response = self.guardrail_llm.invoke(messages)
        if response.content.startswith("VIOLATION"):
            violation_reason = response.content.split(":", 1)[1].strip()
            print("Policy violation detected:", violation_reason)
            state["denial_reason"] = f"Policy violation detected: {violation_reason}"
            return {"denial_reason": f"I'm sorry, but I cannot provide that information. {violation_reason}", "continue_execution": False}
        return { "question": state["question"] }

    def role_checking_agent(self, state: State):
        print("Role checking agent running: Question:", state["question"], state["rbac"])
        if not state["continue_execution"]:
            return {"answer": "I'm sorry, but I cannot provide that information." + state["denial_reason"]}
        role_checking_template = """
            You are a security agent responsible for checking if the question is appropriate for the user's role, based on the role based access control (RBAC) attributes shared below. 

            Determine if this question is appropriate for the user's role and permissions.
            Users with access to HR systems will be allowed to access all employee and personnel information
            Users with access to finance systems will be allowed to access all financial information and related business information. 

            If access should be denied:
            - Return "DENIED: [specific reason why access is not permitted based on role]"

            If access is permitted:
            - Return a list of allowed sensitive information categories based on the user's permissions. Dont give any other information.
            - Format: ["category1", "category2", ...]
            - Only include categories from list_of_sensitive_information_to_check that match the user's permissions

            Question: {question}

            User Role Information:
            {role_json}

            Sensitive Information Categories to Check:
            {list_of_sensitive_information_to_check}
        """
        role_checking_prompt = self.PromptTemplate.from_template(role_checking_template)
        messages = role_checking_prompt.invoke({
            "role_json": state["rbac"],
            "question": state["question"],
            "list_of_sensitive_information_to_check": self.SENSITIVE_INFORMATION_TYPES
        })
        response = self.guardrail_llm.invoke(messages)
        if response.content.startswith("DENIED"):
            print("Access denied by role checking agent:", response.content)
            return {"answer": f"I'm sorry, but I cannot provide that information. {response.content}", "role_check_passed": False, "denial_reason": response.content, "continue_execution": False}
        else:
            print("Access approved by role checking agent")
            return {"role_check_passed": True, "sensitive_information_allowed": response.content}

    def post_processing_agent(self, state: State):
        print("Post processing agent running: sensitive_information_allowed", state["sensitive_information_allowed"])
        sensitive_information_not_allowed = [info_type for info_type in self.SENSITIVE_INFORMATION_TYPES if info_type not in state["sensitive_information_allowed"]]
        print("Post processing agent running: sensitive_information_not_allowed", sensitive_information_not_allowed)
        if not sensitive_information_not_allowed:
            return {"answer": state["answer"]}
        post_processing_agent_template = """
            You are an assistant helping to review responses. 
            
            Information allowed: {sensitive_information_allowed}
            -
            Information not allowed: {sensitive_information_not_allowed}

            
            Follow the policy above strictly. Only if disallowed information is found, reply "I'm sorry, but I cannot provide that information as it contains sensitive information or content that is not allowed to be shared with your role."
            Otherwise, return the original answer only. DONT GENERATE ANYTHING ELSE.

            **IMPORTANT**: Return the original answer only if you are uncertain. Dont generate anything else. 

            ======
            Original Answer:
            {original_answer} 

        """    
        post_processing_prompt = self.PromptTemplate.from_template(post_processing_agent_template)
        messages = post_processing_prompt.invoke({
            "original_answer": state["answer"], 
            "sensitive_information_allowed": state["sensitive_information_allowed"],
            "sensitive_information_not_allowed": sensitive_information_not_allowed
        })
        sanitized_response = self.guardrail_llm.invoke(messages)
        if not sanitized_response.content == state["answer"]:
            print("Sensitive information caught | Original Answer: ", state["answer"])
            print("Sanitized Answer: ", sanitized_response.content)
        return {"answer": sanitized_response.content}

    def graph_builder(self, State=State, Experiment=None):
        seq_to_run = [self.setup]
        if "Auth" in Experiment["guardRails"]:
            seq_to_run += [self.auth_agent]
        if "Alignment" in Experiment["guardRails"]:
            seq_to_run += [self.policy_alignment_agent]
        if "RoleChecking" in Experiment["guardRails"]:
            seq_to_run += [self.role_checking_agent]
        if Experiment["workflow"] == "RAG":
            seq_to_run += [self.retrieve, self.generate]
        elif Experiment["workflow"] == "Chat":
            seq_to_run += [chat]
        if "PostProcessing" in Experiment["guardRails"]:
            seq_to_run += [self.post_processing_agent]
        print("Building graph with: ", [_.__name__ for _ in seq_to_run])
        graph_builder = StateGraph(State).add_sequence(seq_to_run)
        graph_builder.add_edge(START, seq_to_run[0].__name__)
        return graph_builder