from sentence_transformers import CrossEncoder
from collections import Counter
cross_enc = CrossEncoder("nreimers/mmarco-mMiniLMv2-L12-H384-v1",device="cuda",trust_remote_code=True)
from dexter.llms.llm_engine_orchestrator import LLMEngineOrchestrator
import json
import pandas as pd
import torch

from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-large-mnli",force_download=True).cuda()

def extract_answer(generated):
    if '\n' not in generated:
        last_line =  generated
    else: 
        last_line = generated.split('\n')[-1]

    if ':' not in last_line:
        after_colon = last_line
    else:
        after_colon = generated.split(':')[-1]
    
    if ' ' == after_colon[0]:
        after_colon = after_colon[1:]
    if '.' == after_colon[-1]:
        after_colon = after_colon[:-1]

    return after_colon

def extract_question(generated):
    if '\n' not in generated:
        last_line =  generated
    else: 
        last_line = generated.split('\n')[-1]

    if 'Follow up:' not in last_line:
      print('we probably should never get here...' + generated)

    if ':' not in last_line:
        after_colon = last_line
    else:
        after_colon = generated.split(':')[-1]
    
    if ' ' == after_colon[0]:
        after_colon = after_colon[1:]
    if '?' != after_colon[-1]:
        print('we probably should never get here...' + generated)

    return after_colon
def drop_docnos_from_counters(docnos, counters):
        for docno in docnos:
            for c in counters:
                del c[docno]
def self_con(tmp_list):
    ans_list = []
    for tmp in tmp_list:
        # tmp_list.append(compare_llm_outputs(user_query))
        # tmp = compare_llm_outputs(user_query)
        # print(tmp)
        ans = ""
        if len(tmp.split("[Final Answer]:"))>0:
            ans = tmp.split("[Final Answer]:")[-1]
            #print(ans)
            #ans = ans.split("\n")[0
        
       
            ans_list.append(ans)
    # print(ans_list)

    d = {}
    for i in ans_list:
        if len(list(d.keys()))>0:
            for key in list(d.keys()):
                if i.replace(".","").replace(",","") in key.replace(".","").replace(",","") or key.replace(".","").replace(",","") in i.replace(".","").replace(",",""):
                    d[key] += 1
                else:
                    d[i] = 1
        else:
            d[i]=1
    print(d)
    n = sorted(d.items(), key=lambda x:x[1], reverse=True)
    print("n",n)
    return n

def get_answer_consistency(doc_ids,scores,corpus,question,evidence=None):
    config_instance = LLMEngineOrchestrator()
    llm_instance = config_instance.get_llm_engine(data="",llm_class="openai",model_name="gpt-3.5-turbo", top_n=10)
    docs = []
    if evidence is None:
        for did, score in zip(doc_ids, scores):
            docs.append(corpus[int(did)].text())
    else:
        docs = evidence
    system_prompt = "Follow the given examples and Given the question and evidences, think step by step give rationale form your own reasoning path preced by [Answer]: and output final answer for the question using information from given evidences and give concise precise answer preceded by [Final Answer]:\n"
    top_2 = "\n".join(docs)
    try:
        user_prompt = """ 
                [Question]: When does monsoon season end in the state the area code 575 is located?
[Answer]: The area code 575 is located in New Mexico. Monsoon season in New Mexico typically ends in mid-September.
[Final Answer]: mid-September.

Question: Who lived longer, Theodor Haecker or Harry Vaughan Watkins?
[Answer]: Theodor Haecker was 65 years old when he died.Harry Vaughan Watkins was 69 years old when he died. Hence Harry Vaughan Watkins lived longer.
[Final Answer]: Harry Vaughan Watkins.

[Question]: What is the current official currency in the country where Ineabelle Diaz is a citizen?
[Answer]: Ineabelle Diaz is from Peurto Rico, which is in the United States of America. The current official currency in the United
States is the United States dollar. 
[Final Answer]: United States dollar.

[Question]: Where was the person who founded the American Institute of Public Opinion in 1935 born?
[Answer]: The person who founded the American Institute of Public Opinion in 1935 is George Gallup. George Gallup was born
in Jefferson, Iowa. 
[Final Answer]: Jefferson.
[Question]: What language is used by the director of Tiffany Memorandum?
[Answer]: The director of Tiffany Memorandum is Sergio Grieco. Sergio Grieco speaks Italian.
[Final Answer]: Italian.

[Question]: What is the sports team the person played for who scored the first touchdown in Superbowl 1?
[Answer]: The player that scored the first touchdown in Superbowl 1 is Max McGee. Max McGee played for the Green Bay
Packers.
[Final Answer]: Green Bay Packers.
[Question]: The birth country of Jayantha Ketagoda left the British Empire when?
[Answer]: The birth country of Jayantha Ketagoda is Sri Lanka. Sri Lanka left the British Empire on February 4, 1948.
[Final Answer]: February 4, 1948.\n\n """ + "Follow the above example and given"+"the evidence, Evidence: "+top_2+" \n select and use the most relevant information for the question from the given evidences preceded by Evidence: and form your own reasoning thinking step by step preceded by [Answer]: and subsequently give concise final answer as shown in above examples strictly preceded by [Final Answer]: for the Question:"+question
    #print("reasoning","\n".join(decomposition_path),top_3)
        chain_answer = llm_instance.get_chat_completion_without_stop_multiple(user_prompt,system_prompt)
    except:
        top_2 = "\n".join(docs[:20])

        user_prompt = """ 
                [Question]: When does monsoon season end in the state the area code 575 is located?
[Answer]: The area code 575 is located in New Mexico. Monsoon season in New Mexico typically ends in mid-September.
[Final Answer]: mid-September.

Question: Who lived longer, Theodor Haecker or Harry Vaughan Watkins?
[Answer]: Theodor Haecker was 65 years old when he died.Harry Vaughan Watkins was 69 years old when he died. Hence Harry Vaughan Watkins lived longer.
[Final Answer]: Harry Vaughan Watkins.

[Question]: What is the current official currency in the country where Ineabelle Diaz is a citizen?
[Answer]: Ineabelle Diaz is from Peurto Rico, which is in the United States of America. The current official currency in the United
States is the United States dollar. 
[Final Answer]: United States dollar.

[Question]: Where was the person who founded the American Institute of Public Opinion in 1935 born?
[Answer]: The person who founded the American Institute of Public Opinion in 1935 is George Gallup. George Gallup was born
in Jefferson, Iowa. 
[Final Answer]: Jefferson.
[Question]: What language is used by the director of Tiffany Memorandum?
[Answer]: The director of Tiffany Memorandum is Sergio Grieco. Sergio Grieco speaks Italian.
[Final Answer]: Italian.

[Question]: What is the sports team the person played for who scored the first touchdown in Superbowl 1?
[Answer]: The player that scored the first touchdown in Superbowl 1 is Max McGee. Max McGee played for the Green Bay
Packers.
[Final Answer]: Green Bay Packers.
[Question]: The birth country of Jayantha Ketagoda left the British Empire when?
[Answer]: The birth country of Jayantha Ketagoda is Sri Lanka. Sri Lanka left the British Empire on February 4, 1948.
[Final Answer]: February 4, 1948.\n\n """ + "Follow the above example and given"+"the evidence, Evidence: "+top_2+" \n select and use the most relevant information for the question from the given evidences preceded by Evidence: and form your own reasoning thinking step by step preceded by [Answer]: and subsequently give concise final answer as shown in above examples strictly preceded by [Final Answer]: for the Question:"+question
    #print("reasoning","\n".join(decomposition_path),top_3)
        chain_answer = llm_instance.get_chat_completion_without_stop_multiple(user_prompt,system_prompt)
    out_text = []
    for index, ans in enumerate(chain_answer.choices):
        text = ans.message.content
        # if len(text.split("[Final Answer]:"))>0:
        #     ans = text.split("[Final Answer]:")[-1]
        out_text.append(text)
    semantic_set_ids= {}
    deberta_predictions = []
    answer_list_1 = []
    answer_list_2 = []
    has_semantically_different_answers = False
    inputs = []
    syntactic_similarities = {}
    unique_generated_texts = list(set(out_text))
    for index, answer in enumerate(unique_generated_texts):
        semantic_set_ids[answer] = index
    count=0
    if len(unique_generated_texts) > 1:
        
        # Evalauate semantic similarity
        for i, reference_answer in enumerate(unique_generated_texts):
            for j in range(i + 1, len(unique_generated_texts)):

                answer_list_1.append(unique_generated_texts[i])
                answer_list_2.append(unique_generated_texts[j])
                try:
                    qa_1 = question + ' ' + unique_generated_texts[i]
                    qa_2 = question + ' ' + unique_generated_texts[j]
                # qa_1 = unique_generated_texts[i]
                # qa_2 = unique_generated_texts[j]

                    input = qa_1 + ' [SEP] ' + qa_2
                    encoded_input = tokenizer.encode(input, padding=True,max_length=512, truncation=True)
                    prediction = model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
                    predicted_label = torch.argmax(prediction, dim=1)

                    reverse_input = qa_2 + ' [SEP] ' + qa_1
                    encoded_reverse_input = tokenizer.encode(reverse_input, padding=True,max_length=512, truncation=True)
                    reverse_prediction = model(torch.tensor(torch.tensor([encoded_reverse_input]), device='cuda'))['logits']
                    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)
                except:
                    qa_1 = unique_generated_texts[i]
                    qa_2 = unique_generated_texts[j]
                # qa_1 = unique_generated_texts[i]
                # qa_2 = unique_generated_texts[j]

                    input = qa_1 + ' [SEP] ' + qa_2
                    encoded_input = tokenizer.encode(input, padding=True,max_length=512, truncation=True)
                    
                    prediction = model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
                    predicted_label = torch.argmax(prediction, dim=1)

                    reverse_input = qa_2 + ' [SEP] ' + qa_1
                    encoded_reverse_input = tokenizer.encode(reverse_input, padding=True,max_length=512,truncation=True)
                    reverse_prediction = model(torch.tensor(torch.tensor([encoded_reverse_input]), device='cuda'))['logits']
                    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)

                deberta_prediction = 2
                #print(qa_1, qa_2, predicted_label, reverse_predicted_label)
                if 0 in predicted_label or 0 in reverse_predicted_label:
                    has_semantically_different_answers = True
                    count+=1
                    deberta_prediction = 0

                else:
                    semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]

                deberta_predictions.append([unique_generated_texts[i], unique_generated_texts[j], deberta_prediction])
    answer_consistent = None
    consistency_list = self_con(out_text)
    print("count***",count, len(consistency_list))
    if count ==0:
        penalty = 1
    else:
        penalty = count
    if( len(consistency_list) > 2 and count >2):
        answer_consistent=False
    else:
        answer_consistent=True
    return answer_consistent, penalty
       


                
def update_frontier(doc_ids,scored_scores, frontier, frontier_data, scored_dids, knn_neighbours):
        remaining_budget = 100  - len(scored_dids)
        for score, did in sorted(zip(scored_scores, doc_ids), reverse=True):
            if len(frontier) < remaining_budget or score >= frontier_data['minscore']:
                hit = False
                #sorted_neighbours = sorted(zip(list(knn_neighbours[str(did)].keys()),zip(knn_neighbours[str(did)].values())), key=lambda x: x[1],reverse=True)[:20]
                for target_did in knn_neighbours[str(did)]:
                    if target_did not in scored_dids:
                        if target_did not in frontier or score > frontier[target_did]:
                            frontier[target_did] = score
                            hit = True
                if hit and score < frontier_data['minscore']:
                    frontier_data['minscore'] = score
        print("here frontier",score)
def get_answer(response, question_id, corpus, query, knn_neighbours):
    print("rerank **********")
    batch_size = 16
    scores = {}
    response_final = {}
    docs_sorted = sorted(zip(list(response[question_id].keys()), list(response[question_id].values())),key=lambda x: x[1],reverse=True)[:100]
    res_map = [Counter(dict(docs_sorted))]
    frontier_data = {'minscore': float('inf')}
    iteration = 0
    res_map.append(Counter())
    answer_consistent = False
    print("answer_consistent****",answer_consistent)
    while len(scores) < 100  and any(r for r in res_map):
            #print("len(scores)",len(scores), iteration,len(res_map),len(res_map[iteration%len(res_map)]))

            if len(res_map[iteration%len(res_map)]) == 0:
                    # if there's nothing available for the one we select, skip this iteration (i.e., move on to the next one)
                    iteration += 1
                    continue
            this_res = res_map[iteration%len(res_map)]
            size = min(batch_size, 100  - len(scores))
            batch = this_res.most_common(size)
            #print(batch)
            doc_ids = []
            scored_docs = []
            for idx,scored1 in batch:
                #print(idx,scored1)
                corpus_id = int(idx)
                new_corpus = ""
                corpus_text = corpus[corpus_id]
        #print(setfit_model.predict_proba([ queries[index].text() +"[SEP]"+corpus_text.text()]))
        #final_ans = setfit_model.predict_proba([ queries[index].text() +"[SEP]"+corpus_text.text()])[0][1].cpu().numpy()
                final_ans = cross_enc.predict([[query, corpus_text.text()]])[0]
                
                doc_ids.append(idx)
                scored_docs.append(float(final_ans))
            scores.update({k: (s, iteration) for k, s in zip(doc_ids, scored_docs)})
            answer_consistent, penalty = get_answer_consistency(doc_ids, scores, corpus, query)
            scores.update({k: (s/(penalty/2), iteration) for k, s in zip(doc_ids, scored_docs)})
            #drop_docnos_from_counters(doc_ids, res_map)
            for docno in doc_ids:
                for c in res_map:
                    del c[docno]
            if len(scores) < 100:
                    #update_frontier(doc_ids,scored_docs, res_map[1], frontier_data, scores, knn_neighbours)
                    remaining_budget = 100  - len(scores)
                    for score, did in sorted(zip(scored_docs, doc_ids), reverse=True):
                        if len(res_map[1]) < remaining_budget or score >= frontier_data['minscore']:
                            hit = False
                            sorted_neighbours = sorted(zip(list(knn_neighbours[str(did)].keys()),zip(knn_neighbours[str(did)].values())), key=lambda x: x[1],reverse=True)[:10]
                            for target_did, _  in sorted_neighbours:# knn_neighbours[str(did)]:
                                if target_did not in scores:
                                    if target_did not in res_map[1] or score > res_map[1][target_did]:
                                        res_map[1][target_did] = score
                                        hit = True
                            if hit and score < frontier_data['minscore']:
                                frontier_data['minscore'] = score
                    #print("here frontier",score)
            # answer_consistent, penalty = get_answer_consistency(scores, corpus, query)
            # scores.update({k: (s/int(penalty), iteration) for k, (s,i) in zip(list(scores.keys()), list(scores.values()))})
            print(scores,len(doc_ids),len(scores))
            iteration += 1
    scored_dids = []
    scored_scores = []
    for did, (score, i) in Counter(scores).most_common():
        scored_dids.append(did)
        scored_scores.append(score)

    # Backfill unscored items
    # if  len(scores) < 100 :
    #     last_score = scored_scores[-1] if result['score'] else 0.
    #     count = min(100  - len(scores), len(res_map[0]))
    #     for i, (did, score) in enumerate(res_map[0].most_common()):
    #         if i >= count:
    #             break
    #         scored_dids.append(did)
    #         scored_scores.append(last_score - 1 - i)
        # top_neighboirs = sorted(zip(list(knn_neighbours[str(corpus_id)].keys()),list(knn_neighbours[str(corpus_id)].values())), key=lambda x: x[1],reverse=True)[:40]
        # neighbours = [node for node, score in top_neighboirs]#list(knn_neighbours[str(corpus_id)].keys())
        # for neighbor in neighbours:
        #     neighbour_id = int(neighbor)
        #     corpus_text_1 = corpus[neighbour_id]
        #     #print("setfit_model.predict_proba([ queries[index].text() +corpus_text_1.text()])",setfit_model.predict_proba([ queries[index].text() +"[SEP]"+corpus_text_1.text()]))
        #     if neighbour_id not in doc_ids:
        #             final_ans = cross_enc.predict([ [query ,corpus_text_1.text()]])[0]
        #            # print("****))",final_ans,knn_neighbours[str(corpus_id)][neighbour_id])
        #             doc_ids.append(neighbour_id)
        #             scores.append(float(final_ans))

    re_ranked_list = sorted(zip(list(scored_dids),list(scored_scores)), key=lambda x: x[1],reverse=True)[:10]
    re_ranked_final = []
    running_list = []
    for re_id, re_score in re_ranked_list:
        running_list.append(corpus[int(re_id)].text())
        # answer_consistent = get_answer_consistency(None, corpus,query,evidence=running_list)
        # if answer_consistent:
        #     re_ranked_final.append(corpus[int(re_id)].text())
        # else:
        re_ranked_final.append(corpus[int(re_id)].text())
    print("re_ranked_final",len(re_ranked_final), len(scored_dids))
    return re_ranked_final,penalty

def get_last_line(generated):
    if '\n' not in generated:
        last_line =  generated
    else: 
        last_line = generated.split('\n')[-1]


    return last_line

if __name__=="__main__":
        config_instance = LLMEngineOrchestrator()
        llm_instance = config_instance.get_llm_engine(data="",llm_class="openai",model_name="gpt-3.5-turbo")
        #assertTrue(isinstance(llm_instance, OpenAIEngine))
        with open("data/intermediate_outputs/wqa_splade_docs.json") as f:
                evidence = json.load(f)
        with open("data/intermediate_outputs/wqa_splade.json","r") as f:
            response = json.load(f)
        with open("data/intermediate_outputs/gar_graph.json", "r") as f:
            knn_neighbours = json.load(f)
        question_df = {"questions":[],"answers":[], "reasoning_path":[],"meta_reasoner":[]}

        loader = RetrieverDataset("wikimultihopqa","wiki-musiqueqa-corpus","evaluation/config.ini",Split.DEV)
        queries, qrels, corpus = loader.qrels()
        raw_data = loader.base_dataset.raw_data
        matches = 0
        mismatches = 0
        intermediate = "\nIntermediate Answer:"
        ids = []
        responses_final = []
        for row in raw_data:
                system_prompt = "Follow the given examples \n"
                response_query = {}
                if row.question.id() in ids:
                        continue
                else:
                        ids.append(row.question.id())
                followup = "Follow up:"
                finalans= '\n[Final Answer]:'
                

                cur_prompt = """Question: Who lived longer, Theodor Haecker or Harry Vaughan Watkins?
                Are follow up questions needed here: Yes.
Follow up: How old was Theodor Haecker when he died?
Intermediate Answer: Theodor Haecker was 65 years old when he died.
Follow up: How old was Harry Vaughan Watkins when he died?
Intermediate Answer: Harry Vaughan Watkins was 69 years old when he died.
[Final Answer]: Harry Vaughan Watkins.

Question: Why did the founder of Versus die?
Are follow up questions needed here: Yes.
Follow up: Who founded Versus?
Intermediate Answer: Gianni Versace.
Follow up: Why did Gianni Versace die?
Intermediate Answer: Gianni Versace was shot and killed on the steps of his Miami Beach mansion on July 15, 1997.
[Final Answer]: Shot.

Question: Who is the grandchild of Dambar Shah?
Are follow up questions needed here: Yes.
Follow up: Who is the child of Dambar Shah?
Intermediate Answer: Dambar Shah (? - 1645) was the king of the Gorkha Kingdom. He was the father of Krishna Shah.
Follow up: Who is the child of Krishna Shah?
Intermediate Answer: Krishna Shah (? - 1661) was the king of the Gorkha Kingdom. He was the father of Rudra Shah.
[Final Answer]: Rudra Shah.

Question: Who was born earlier director of Avengers or director of Frankenstein (1931 film)
Follow up: Who is the director of avengers?
Intermediate Answer: Russo brothers
Follow up: Who is the director of Frankenstein (1931 film)?
Intermediate Answer: James Whale
Follow up: Who was born earlier Russo brothers or James Whale?
Intermediate Answer: Russo brothers
[Final Answer]: Avengers

Question: Are both director of film FAQ: Frequently Asked Questions and director of film The Big Money from the same
country?
Are follow up questions needed here: Yes.
Follow up: Who directed the film FAQ: Frequently Asked Questions?
Intermediate Answer: Carlos Atanes.
Follow up: Who directed the film The Big Money?
Intermediate Answer: John Paddy Carstairs.
Follow up: What is the nationality of Carlos Atanes?
Intermediate Answer: Carlos Atanes is Spanish.
Follow up: What is the nationality of John Paddy Carstairs?
Intermediate Answer: John Paddy Carstairs is British.
[Final Answer]: No.

Question: Are both the directors of Jaws and Casino Royale from the same country? 
Are follow up questions needed here: Yes. 
Follow up: Who is the director of Jaws? 
Intermediate Answer: The director of Jaws is Steven Spielberg. 
Follow up: Where is Steven Spielberg from? 
Intermediate Answer: The United States. 
Follow up: Who is the director of Casino Royale? 
Intermediate Answer: The director of Casino Royale is Martin Campbell. 
Follow up: Where is Martin Campbell from? 
Intermediate Answer: New Zealand. 
[Final Answer]: No

Question: """ +row.question.text() + "Are follow up questions needed here: "
                ret_text = llm_instance.get_chat_completion(cur_prompt,system_prompt, intermediate)
                print("ret_textret_text**",ret_text)
                ret_text = llm_instance.get_chat_completion(cur_prompt+"Yes",system_prompt, intermediate)
                print("ret_textret_text2**",ret_text)
                docs_this_round =[]
                decomposition_path = []
                while followup in get_last_line(ret_text):  
                    print("here*** followup",followup)        
                    cur_prompt += ret_text
                    question = extract_question(ret_text)
                    print("question****)))",question)
                    external_answer,consistency_list = get_answer(response, str(row.question.id()), corpus, question, knn_neighbours)

                    if external_answer is not None:
                        
                        try:
                            external_answer_1 = "\n".join(external_answer)
                            #cur_prompt += "Evidence:"+external_answer + '\nGiven evidence generate only the Intermediate Answer:'
                            prompt_1 = "Follow up: "+question + " Evidence:"+external_answer_1 + '\nGiven the question and related evidencet think step by step and generate the Intermediate Answer:'
                            interim= "Intermediate Answer: "+llm_instance.get_chat_completion(prompt_1,system_prompt,"\n")
                            decomposition_path.append("Question:"+question)
                            decomposition_path.append("\nIntermediate Answer:"+interim)
                        except:
                            external_answer_1 = "\n".join(external_answer[:8])
                            #cur_prompt += "Evidence:"+external_answer + '\nGiven evidence generate only the Intermediate Answer:'
                            prompt_1 = "Follow up: "+question + " Evidence:"+external_answer_1 + '\nGiven the question and related evidencet think step by step and generate the Intermediate Answer:'
                            interim= "Intermediate Answer: "+llm_instance.get_chat_completion(prompt_1,system_prompt,"\n")
                            decomposition_path.append("Question:"+question)
                            decomposition_path.append("\nIntermediate Answer:"+interim)
                        print("\ninterim**",interim + ' '  + '.', end='' )
                        for doc1 in external_answer[:2]:
                            docs_this_round.append(doc1)
                        cur_prompt+=interim+'.'
                        ret_text = llm_instance.get_chat_completion(cur_prompt,system_prompt, intermediate)
                        #decomposition_path.append("\[Final Answer]:"+interim)

                        print("\nexternal answer",ret_text)
                    else:
                        #We only get here in the very rare case that Google returns no answer.
                        cur_prompt += intermediate
                        print(intermediate + ' ')
                        gpt_answer = llm_instance.get_chat_completion(cur_prompt,system_prompt, ['\n'+followup, finalans])
                        cur_prompt += gpt_answer

                    
                if "[Final Answer]:" not in ret_text:
                    cur_prompt += finalans
                    print(finalans, end = '')
                    ret_text = llm_instance.get_chat_completion(cur_prompt,system_prompt,"\n")
                #print("user_prompt",user_prompt)
                #chain_answer = llm_instance.get_chat_completion(user_prompt,system_prompt)
                llm_instance_1 = config_instance.get_llm_engine(data="",llm_class="openai",model_name="gpt-3.5-turbo",top_n=10)
                self_con_inputs = []
        #         self_con_input = llm_instance_1.get_chat_completion_without_stop_multiple(cur_prompt.split(finalans)[0]+finalans,system_prompt)
        #         for index, ans in enumerate(self_con_input.choices):
        #             text = ans.message.content
        # # if len(text.split("[Final Answer]:"))>0:
        # #     ans = text.split("[Final Answer]:")[-1]
        #             self_con_inputs.append(text)
        #         print("self_con_input**",self_con_input)
        #         self_con_output = self_con(self_con_inputs)
                if len(ret_text.split("[Final Answer]:")) >0:
                    final_ans = ret_text.split("[Final Answer]:")[-1]# extract_answer(cur_prompt+ret_text)
                elif len(ret_text.split("answer is:")) >1:
                    final_ans = ret_text.split("answer is:")[-1]
                print("final_ans+",final_ans)
                #intermediate_1_query = intermediate_1.split("Follow up:")[-1]
                print("docs_this_round",len(docs_this_round))
                final_docs = []
                final_scores = []
                for doc in docs_this_round:
                    final_score = cross_enc.predict([[row.question.text(), doc]])[0]
                    final_docs.append(doc)
                    final_scores.append(final_score)
                re_ranked_list_final = sorted(zip(list(final_docs),list(final_scores)), key=lambda x: x[1],reverse=True)[:10]
                docs_final = [document for document,score in re_ranked_list_final]
                print("docs_final",len(docs_final), docs_final[0])
                top_final = "\n".join(docs_final)
                    
                #top_final = "\n".join(list(set(docs_this_round)))
                system_prompt_1 = "Follow the given examples and Given the question and context, reasoning path, think step by step extract key segments from given evidence relevant to question and give rationale, by forming your own reasoning path preceded by [Answer]: and output final answer for the question using information from given evidences and give concise precise answer preceded by [Final Answer]:\n"

                user_prompt = """ 
                    [Question]: When does monsoon season end in the state the area code 575 is located?
    [Answer]: The area code 575 is located in New Mexico. Monsoon season in New Mexico typically ends in mid-September.
    [Final Answer]: mid-September.

    Question: Who lived longer, Theodor Haecker or Harry Vaughan Watkins?
    [Answer]: Theodor Haecker was 65 years old when he died.Harry Vaughan Watkins was 69 years old when he died. Hence Harry Vaughan Watkins lived longer.
    [Final Answer]: Harry Vaughan Watkins.

    [Question]: What is the current official currency in the country where Ineabelle Diaz is a citizen?
    [Answer]: Ineabelle Diaz is from Peurto Rico, which is in the United States of America. The current official currency in the United
    States is the United States dollar. 
    [Final Answer]: United States dollar.

    [Question]: Where was the person who founded the American Institute of Public Opinion in 1935 born?
    [Answer]: The person who founded the American Institute of Public Opinion in 1935 is George Gallup. George Gallup was born
    in Jefferson, Iowa. 
    [Final Answer]: Jefferson.
    [Question]: What language is used by the director of Tiffany Memorandum?
    [Answer]: The director of Tiffany Memorandum is Sergio Grieco. Sergio Grieco speaks Italian.
    [Final Answer]: Italian.

    [Question]: What is the sports team the person played for who scored the first touchdown in Superbowl 1?
    [Answer]: The player that scored the first touchdown in Superbowl 1 is Max McGee. Max McGee played for the Green Bay
    Packers.
    [Final Answer]: Green Bay Packers.

    [Question]: The birth country of Jayantha Ketagoda left the British Empire when?
    [Answer]: The birth country of Jayantha Ketagoda is Sri Lanka. Sri Lanka left the British Empire on February 4, 1948.
    [Final Answer]: February 4, 1948. \n""" + "Follow the above example and Given existing Reasoning path: "+"\n".join(decomposition_path)+"the evidence, Evidence: "+top_final+" \n and use the most relevant information for the question from the most relevant evidence from given Evidence: and form your own correct reasoning path to derive the answer thinking step by step preceded by [Answer]: and subsequently give final answer as shown in above examples preceded by [Final Answer]: for the Question:"+row.question.text()

                    #print("reasoning","\n".join(decomposition_path),top_3)
                chain_answer = llm_instance.get_chat_completion_without_stop(user_prompt,system_prompt_1)
               # multiple_chain_answer = llm_instance_1.get_chat_completion_without_stop_multiple(user_prompt,system_prompt_1)

        #         for index, ans in enumerate(multiple_chain_answer.choices):
        #             text = ans.message.content
        # # if len(text.split("[Final Answer]:"))>0:
        # #     ans = text.split("[Final Answer]:")[-1]
        #             self_con_inputs.append(text)
        #         self_con_output = self_con(self_con_inputs)
        #         #final_ans = final_ans.lower()
        #         if self_con_output[0][1] >=1 or "not possible" in final_ans or "cannot" in final_ans or "inconclusive" in final_ans or "unknown" in final_ans:
        #             for idx2,_ in enumerate(self_con_output):
        #                 if "not possible" in self_con_output[idx2][0].lower() or "unknown" in self_con_output[idx2][0].lower() or "cannot" in self_con_output[idx2][0].lower():
        #                     continue
        #                 else:
        #                     chain_answer = self_con_output[idx2][0]
        #                     break
        #             #chain_answer = self_con_output[0][0]
        #         else:
        #             chain_answer = final_ans
                if "not possible" in chain_answer.lower():
                        mismatches+=1
                        
                elif "unknown" in chain_answer.lower():
                        mismatches+=1
                        
                if len(chain_answer.split("[Final Answer]:")) >0:
                    chain_answer = chain_answer.split("[Final Answer]:")[-1]# extract_answer(cur_prompt+ret_text)
                elif len(chain_answer.split("answer is:")) >0:
                    chain_answer = chain_answer.split("answer is:")[-1]
                # else:
                #     chain_answer = self_con_output[0][0]
                print("final_ans+",chain_answer)
                # elif len(chain_answer.split("[Final Answer]:")) >0:
                #         answer = chain_answer.split("[Final Answer]:")[-1]
                print("************",chain_answer,row.answer.text())
                if row.answer.text().lower().replace(",","").replace("-","").replace("–","") in chain_answer.lower().replace(",","").replace("-","").replace("–",""):
                            matches+=1
                else:
                            mismatches+=1
                question_df["answers"].append(final_ans)
                question_df["questions"].append(row.question.text())
                question_df["reasoning_path"].append(cur_prompt)
                question_df["meta_reasoner"].append(chain_answer)

                response_query["query"] = row.question.text()
                response_query["evidences"] = []
                for evid in docs_this_round:
                    response_query["evidences"].append(evid)
                response_query["meta_reasoner_answer"]= chain_answer
                response_query["sequential_reasoner_answer"]= ret_text
                response_query["sequential_reasoner_path"]= cur_prompt
                response_query["correct_answer"]= row.answer.text()

                responses_final.append(response_query)

                final_questions = pd.DataFrame(question_df)
                print("EM", matches/(matches+mismatches))
                print(final_questions)
                final_questions.to_csv("sunar_wqa.tsv",sep="\t",index=False)
                with open("sunar_wqa.json","w") as f:
                    json.dump(responses_final,f)