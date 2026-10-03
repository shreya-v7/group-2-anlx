# **Welcome to LLM Box -- a software application to facilitate low-cost, low-compute (no GPU) customation of LLMs!**

This application was designed by Dr. Sara Kingsley (with coding Assistance from various LLMs). The idea and design are original works of the author.

## **University Implementations**
This version of LLMBox was made for Dr. Kingsley's `Generative AI` university courses. 

* For **general instructions**, please see below. 
* For **course-specific instructions**, please see the `about` directory.

## **Accetable Uses of LLMBox**
Please adhere to these acceptable use guidelines for LLMBox:

* You may use and modify LLMBox code. Please attribute the software to Dr. Kingsley. See the `Licensing Requirements`'` below.
* If you are a university student in one of Professor Kingsley's courses, you may only use university approved software with the LLMBox application.

## **Terms of Use Agreement**
If you use LLMBox, you agree to the following terms and conditions:
* If you use the LLMBox app, you do so at your own risk, and agree to abide by all applicable institution, local, state and federal rules (law).
* You agree not to hold Dr. Sara Kingsley liable for any risks or damages associated with using LLMBox.
* You agree to following all the licensing requirements of LLMBox -- including attribution.
* You agree to following all the licensing requirements of any AI model you use with LLMBox.
* You agree not to engage in prohibited uses of LLMBox. Prohibited uses include but are not limited to those listed in the `Prohibited Uses of LLMBox` section of this document.

### **Prohibited Uses of LLMBox**
Prohibited uses of LLMBox include but are not limited to the following:

* You are prohibited from selling or re-selling for profit any version of LLMBox --- including any modified version of the software you make.
* You are prohibited from using LLMBox to violate any organization, state, local or federal government policy, rule or law.
* You are prohibited from using LLMBox to engage in activities that harm people or animals.
* You are prohibited from using LLMBox to engage in activities that harm or destroy environments (beyond the baseline damage computing does to the environment).
* You are prohibited from distributing this software application to persons who are less than age 18 years.

# **LLMBOX Application Diagram**
```
LLMBOX_APP/
│
├──README.md
├──LICENSE  
├──startllm.py                     # run this in terminal to start the LLM 
├──structure.py                    # creates pydantic models
├──transform_original_dataset.py   # hint: you may use this to transform your dataset into a format accepted by LLMBOX
├──__init__.py
│
├──about/
│     ├── quickstart.md
      ├── LLMBOX_API/               # LLMBOX API DOCUMENTATION
│     └── university_students.md    # forthcoming
│
├──config/   # application configuration manager
│    ├── data/
│    ├── mode/
│    ├── model/
│    ├── optimizer/
│    └── config.yaml
│ 
├──demos/  
│ 
├──finetuned/  
│ 
├──outputs/
│     ├── config_logs/    
│     ├── chat_logs/           # note: you can create training data 
│     ├── test.jsonl                   with the chatlogs.
│     └── train.jsonl
│ 
├──src/             # main application software directory
│    ├── __init__.py
│    ├── datasets.py
│    ├── generation.py
│    ├── modes.py
│    ├── schema.py
│    ├── sys_logger.py
│    ├── tools.py
│    ├──   downloader.py   # use to add LLM models to application
│    └──pydantic_models/
│          ├── personrecords.py
│          └── dogbreeds.py
│
├──models/
│    └───llms/       #place model checkpoints in these directories
│        ├── google/
│        │    └── gemma-3-270m-it/ 
│        ├── microsoft/
│        │   └── phi-4-mini-instruct/
│        ├── cohere/
│        │   └── tiny-aya-global/
│        └── meta/
│            └── llama-3-1-instruct/
│
└──data/  
      ├──demos/
      │
      ├──evaluations/                 # automatically created by LLMBOX
      │
      ├──pydantic_models/             # you need to create these by running relevant LLMBOX modules AFTER defining a Pydantic model in python
      │
      ├──traindata/                   # automatically created by LLMBOX
      │
      ├──transformed_datasets/        # folder Prof. Kingsley made. Feel free to put your transformed original dataset here or elsewhere. 
      │
      │ ### OPTIONAL:
      └──YOUR_DATA/ #place your DATASET in a subdirectory within the data dir.
            ├── original_dataset.json 
            └── dataset_splits/            
                   ├── train_dataset.json  ## split your dataset into train
                   └── test.json           ## & test sets. 
            
```





# **Attribution Statement | How to cite this LLM Software Application:**
If you use, modify or distribute this lab material, you must properly cite or credit the author.

*Recommended citation:*
```Sara Kingsley. September 2026. LLMBox.```

## **License**

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

### **Attribution Requirements**
GPL-3.0 requires that copyright notices, license notices, and existing attribution statements be retained when redistributing the software or derivative works. If you modify the software, you must clearly indicate that changes were made. The original authors' copyright information must not be removed.

### **What GPL-3.0 Requires**

If you distribute this software or a modified version of it, you must:

- Provide a copy of the GPL-3.0 license with the distribution.
- Make the complete corresponding source code available to recipients.
- License any modifications or derivative works under GPL-3.0 as well.
- Clearly document any changes you make to the original code.
- Preserve existing copyright notices and license notices.
- Provide recipients with the same rights to use, study, modify, and redistribute the software that you received.

### **Additional Notes**

- Commercial use is permitted.
- Private/internal use does not require source code disclosure.
- The software is provided **without warranty** or liability.
- You may not impose additional restrictions that limit the rights granted by the GPL-3.0 license.

For the full license text, see the [LICENSE](le or visit the GNU Project website: https://www.gnu.org/licenses/gpl-3.0.en.html. 【1-d432b6】
