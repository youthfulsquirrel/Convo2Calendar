from torch import cuda, bfloat16
import transformers
from huggingface_hub import notebook_login
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.llms import OpenAI
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
import os
import torch

notebook_login()

model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

with open(r'../keys/archive_note.txt', 'r') as fp:
    # read all lines using readline()
    lines = fp.readlines()
    for line in lines:
        os.environ['OPENAI_API_KEY'] = line

def valid_brackets(mystr):
    print('original___')
    print(mystr)
    print('end')
    if mystr[::-1].index(']') < mystr[::-1].index('}') and mystr.index('[') < mystr.index('{'):
        print('string is invalid')
        #newstr = mystr[:mystr.index('[')] + '{' + mystr[mystr.index('[')+1:len(mystr)-mystr[::-1].index(']')-1] + '}' + mystr[len(mystr)-mystr[::-1].index(']'):]
        #print('attempt to correct:', newstr)
        #return(newstr)
        return(False)
    else:
        print('valid')
        
        return(True)
    
def str_contains_brackets(str1):
    value = min(str1.find('{'), str1.find('}'),str1.find('['),str1.find(']'))
    print(value)
    return value != -1

def model_setup():
# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    # begin initializing HF items, you need an access token
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,)


    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto',
    )

    # enable evaluation mode to allow model inference
    model.eval()

    print(f"Model loaded on {device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
    )

    stop_list = ['\nHuman:', '\n```\n']

    stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
    stop_token_ids

    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
    stop_token_ids

    

    # define custom stopping criteria object
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_ids in stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
            return False

    stopping_criteria = StoppingCriteriaList([StopOnTokens()])

    generate_text = transformers.pipeline(
        model=model, 
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        stopping_criteria=stopping_criteria,  # without this model rambles during chat
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # max number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )
    return(generate_text)

def load_transcript():
    with open("transcripts/meeting002.txt") as f:
        transcript_str = f.read()
    return (transcript_str)

def meeting_info(transcript_str):
    meeting_date = ResponseSchema(
            name="meeting_date",
            description="date of the meeting stored in datetime format DD/MM/YYYY.",
        )
    attendees_list = ResponseSchema(
            name="attendees_list",
            description="Full name of everyone present in the meeting, each stored as a string in the list.",
        )

    start_time = ResponseSchema(
            name="start_time",
            description="time the meeting started in 24 hour HH:mm format in datetime",
        )

    end_time = ResponseSchema(
            name="end_time",
            description="time the meeting ended in 24 hour HH:mm format in datetime",
        )

    output_parser = StructuredOutputParser.from_response_schemas(
        [meeting_date, attendees_list, start_time, end_time]
    )

    response_format = output_parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_template("You are a helpful formatting assistant. Return the meeting_date, attendees_list, start_time, end_time separately.        '''{meeting_info}''' \n {format_instructions}")

    llm_openai = OpenAI()
    formated_prompt = prompt.format(**{"meeting_info":transcript_str, "format_instructions":output_parser.get_format_instructions()})
    response_openai = llm_openai(formated_prompt)
    # print(response_openai)
    # print('printing response')
    meeting_info_dict = output_parser.parse(response_openai)
    return(meeting_info_dict)

def meeting_summary(generate_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(TextLoader("transcripts/meeting002.txt").load())

    model_name = ["sentence-transformers/all-mpnet-base-v2", 'sentence-transformers/all-MiniLM-L6-v2', "sentence-transformers/paraphrase-MiniLM-L6-v2"]
    embeddings = HuggingFaceEmbeddings(model_name=model_name[1],model_kwargs={'device': 'cuda'})
    vectorstore=FAISS.from_documents(text_chunks, embeddings)

    llm=HuggingFacePipeline(pipeline=generate_text, model_kwargs={'temperature':0})
    chain =  RetrievalQA.from_chain_type(llm=llm, chain_type = "stuff",return_source_documents=False, retriever=vectorstore.as_retriever())
    query_list = [ "you are a helpful AI meeting minute writer. Help me to write a meeting summary in prose, detailing all the important tasks mentioned, especially the people in charge of each task and the respective deadlines if any. Your minute should be ordered based on the topics discussed and not on chronological order."]
    chat_history = []
    for query in query_list:
        result = chain({"query": query, "chat_history": chat_history})
        #chat_history.append((query, result["result"].strip("\n")))
        result=chain({"query": query, "chat_history": []},return_only_outputs=True)
        print(result['result'])
    return (result['result'])

def meeting_details(transcript_str):
    main_topics = ResponseSchema(
            name="task_description",
            description="Describe each task as detailed as possible and store it as a string in the list.",
        )
    deadline = ResponseSchema(
            name="deadline",
            description="The deadline of the corresponding task_description, stored as a unique string in the list, in the same order as it appears in task_description.",
        )

    person_in_charge = ResponseSchema(
            name="person_in_charge",
            description="default str is NA. The people in involved in each task, who needs to perform follow-up actions after the meeting or had presented this task in the meeting, stored as a string in the list, in the same order as it appears in task_description.",
        )

    days_left = ResponseSchema(
            name="days_left",
            description="Optional parameter, default value is int 0. If the deadline given is relative from the date of the meeting, store as an int (the number of days given) in the list, in the same order as it appears in task_description.",
        )

    output_parser = StructuredOutputParser.from_response_schemas(
        [main_topics, deadline, person_in_charge, days_left]
    )

    response_format = output_parser.get_format_instructions()
    print(response_format)

    prompt = ChatPromptTemplate.from_template("You are a helpful formatting assistant. The meeting transcript is delimited with triple backticks. It is already a summarised version, hence you don't need to summarise it further when providing task_description. A task can be what team members have presented in the meeting, or follow-up actions required after this meeting. Return the task_description, and their respective deadline, person_in_charge and days_left separately, each as a list. '''{meeting_transcript}''' \n {format_instructions}")


    llm_openai = OpenAI()
    meeting_transcript = transcript_str

    formated_prompt = prompt.format(**{"meeting_transcript":meeting_transcript, "format_instructions":output_parser.get_format_instructions()})
    response_openai = llm_openai(formated_prompt)
    #print(response_openai[:5])
    #print(type(response_openai))
    counter = 0
    while not str_contains_brackets(response_openai) or not valid_brackets(response_openai):
        response_openai = llm_openai(formated_prompt)
        counter += 1
        if counter == 10:
            print('already tried 10 times :(')
    formatted_output = output_parser.parse(response_openai)
    return (formatted_output)

def generate_table_content (meeting_info_dict,formatted_output):
    table_content = []
    for i in range (len(formatted_output['task_description'])):
        mydict = {}
        mydict['task_description'] = formatted_output['task_description'][i]
        info_list = [formatted_output['deadline'][i],formatted_output['person_in_charge'][i]]
        mydict['additional_info'] = info_list
        table_content.append(mydict)
    context = meeting_info_dict.copy()
    context['attendees_list'] =  '\n'.join(meeting_info_dict['attendees_list'])
    context['col_labels'] = ['Deadline', 'Person-In-Charge']
    context['tbl_contents'] = table_content
    return(context)



if __name__ == "__main__":
    print('this is main')
    pipeline = model_setup()
    print('loading... 1/6 completed')
    transcript_str = load_transcript()
    print('loading... 2/6 completed')
    meeting_info_dict = meeting_info(transcript_str)
    print('loading... 3/6 completed')
    meeting_summary_text = meeting_summary(pipeline)
    print('loading... 4/6 completed')
    formatted_output = meeting_details(meeting_summary_text)
    print('loading... 5/6 completed')
    context = generate_table_content (meeting_info_dict,formatted_output)
    print('loading... 6/6 completed')
    print(context)