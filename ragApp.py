from typing import Any, Dict, List, Tuple

import requests


class RagApp:
    # Add to Config class
    def answer(self, query: str, include_images: bool = True) -> Tuple[str, List[Dict[str, Any]]]:
            """Answer a query using RAG with improved validation"""
            try:
                # Retrieve relevant information
                # retrieval_results = self.retrieve(query, images=include_images)
                
                # # Calculate context adequacy with debug logging
                # total_context = sum(len(chunk.text) for chunk in retrieval_results.chunks)
                # print(f"Retrieved context length: {total_context} characters")
                
                # # Generate base response if any context exists
                # if retrieval_results.chunks:
                #     print('retrieveal results')
                #     response = self.generate(query, retrieval_results)
                #     if self._validate_response(response, retrieval_results):
                #         return response, self._process_figures(retrieval_results)
                
                # # Fallback only if enabled and no context
                # if Config.FALLBACK_ENABLED:
                #     print("Falling back to external LLM")
                prompt = f"""**Take Context from Recent Research on Alzheimers Disease**


                **Patient Question:**
                {query}

                Please provide a compassionate, evidence-based response as an Alzheimer's specialist and an Assistant:"""

                payload = {
                            "messages":[
                                {
                                    "role":"user",
                                    "content":prompt
                                }
                            ],
                            "model": "meta-llama/llama-3.1-8b-instruct"
                        
                        }
                print('wait')
                response = requests.post(
                            "https://router.huggingface.co/novita/v3/openai/chat/completions",
                            headers={"Authorization": "Bearer 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"},
                            json=payload
                        )
                # print(response.status_code)
                # # if response.status_code == 200:
                #             # Clean up any residual instructions from response
                # print(response)
                # print(response.json())
                                # print(response.json["choices"][0]["message"]['content'])
                full_response = response.json()
                res=full_response['choices'][0]['message']['content']
                
                                # Remove any remaining system messages
                # clean_response = full_response.split("Please provide")[0].strip()
                return res,[]
                # return self._openai_fallback(query), []
                
                # return "I couldn't find relevant research. Please try rephrasing your question.", []
            
            except Exception as e:
                print(f"Critical error: {str(e)}")
                return self._safe_fallback_response(query), []
