# Arabic Synonym BERT-based Adversarial Examples for Text Classification

In this repository, we share code scripts of our synonym BERT-based adversarial attack, our trained models, our adversarial training experiments, and our experiments examining the transferability of attacks between the models, of our accepted paper, titled **Arabic Synonym BERT-based Adversarial Examples for Text Classification**, at *the 18th Conference of the European Chapter of the Association for Computational Linguistics: Student Research Workshop  (EACL 2024 SRW)*, March 21-22, 2024, Malta. 

## Abstract:


Text classification systems have been proven vulnerable to adversarial text examples, modified versions of the original text examples that are often unnoticed by human eyes, yet can force text classification models to alter their classification. Often, research works quantifying the impact of adversarial text attacks have been applied only to models trained in English. In this paper, we introduce the first word-level study of adversarial attacks in Arabic. Specifically, we use a synonym (word-level) attack using a Masked Language Modeling (MLM) task with a BERT model in a black-box setting to assess the robustness of the state-of-the-art text classification models to adversarial attacks in Arabic. To evaluate the grammatical and semantic similarities of the newly produced adversarial examples using our synonym BERT-based attack, we invite four human evaluators to assess and compare the produced adversarial examples with their original examples. We also study the transferability of these newly produced Arabic adversarial examples to various models and investigate the effectiveness of defense mechanisms against these adversarial examples on the BERT models. We find that fine-tuned BERT models were more susceptible to our synonym attacks than the other Deep Neural Networks (DNN) models like WordCNN and WordLSTM we trained. We also find that fine-tuned BERT models were more susceptible to transferred attacks. We, lastly, find that fine-tuned BERT models successfully regain at least 2% in accuracy after applying adversarial training as an initial defense mechanism.


|    Datasets     |     Models    |      Hugging Face Link     |
|---------------- | ------------- | -------------- |
|     **`HARD`**    |    **`WordCNN`**   | [https://huggingface.co/NorahAlshahrani/2dCNNhard](https://huggingface.co/NorahAlshahrani/2dCNNhard)  |
|     **`HARD`**    |   **`WordLSTM`**    | [https://huggingface.co/NorahAlshahrani/biLSTMhard](https://huggingface.co/NorahAlshahrani/biLSTMhard)|
|     **`HARD`**    |     **`BERT`**      | [https://huggingface.co/NorahAlshahrani/BERThard](https://huggingface.co/NorahAlshahrani/BERThard)    |
|**`HARD + Adversarial Examples`**|   **`BERT`**  |[https://huggingface.co/NorahAlshahrani/Adv_BERT_Hard](https://huggingface.co/NorahAlshahrani/Adv_BERT_Hard) |
|     **`MSDA`**    |   **`WordCNN`**     | [https://huggingface.co/NorahAlshahrani/2dCNNmsda](https://huggingface.co/NorahAlshahrani/2dCNNmsda)  |           
|     **`MSDA`**    |   **`WordLSTM`**    | [https://huggingface.co/NorahAlshahrani/biLSTMmsda](https://huggingface.co/NorahAlshahrani/biLSTMmsda)|
|     **`MSDA`**    |     **`BERT`**      | [https://huggingface.co/NorahAlshahrani/BERTmsda](https://huggingface.co/NorahAlshahrani/BERTmsda)    |
|**`MSDA + Adversarial Examples`**|   **`BERT`**  | [https://huggingface.co/NorahAlshahrani/Adv_BERT_msda](https://huggingface.co/NorahAlshahrani/Adv_BERT_msda) |


## BibTeX Citation:
```bash
@inproceedings{alshahrani-etal-2024-bert-synonym-attack,
    title = "{Arabic Synonym BERT-based Adversarial Examples for Text Classification}",
    author = "Alshahrani, Norah  and Alshahrani, Saied  and Wali, Esma  and Matthews, Jeanna ",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: Student Research Workshop",
    month = March,
    year = "2024",
    address = "Malta",
    publisher = "Association for Computational Linguistics",
    abstract = "Text classification systems have been proven vulnerable to adversarial text examples, modified versions of the original text examples that are often unnoticed by human eyes, yet can force text classification models to alter their classification. Often, research works quantifying the impact of adversarial text attacks have been applied only to models trained in English. In this paper, we introduce the first word-level study of adversarial attacks in Arabic. Specifically, we use a synonym (word-level) attack using a Masked Language Modeling (MLM) task with a BERT model in a black-box setting to assess the robustness of the state-of-the-art text classification models to adversarial attacks in Arabic. To evaluate the grammatical and semantic similarities of the newly produced adversarial examples using our synonym BERT-based attack, we invite four human evaluators to assess and compare the produced adversarial examples with their original examples. We also study the transferability of these newly produced Arabic adversarial examples to various models and investigate the effectiveness of defense mechanisms against these adversarial examples on the BERT models. We find that fine-tuned BERT models were more susceptible to our synonym attacks than the other Deep Neural Networks (DNN) models like WordCNN and WordLSTM we trained. We also find that fine-tuned BERT models were more susceptible to transferred attacks. We, lastly, find that fine-tuned BERT models successfully regain at least 2% in accuracy after applying adversarial training as an initial defense mechanism.",
} 
```
