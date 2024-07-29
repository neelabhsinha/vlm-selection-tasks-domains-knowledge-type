knowledge_type_prompt = (
    "Following are the names and explanation of types of knowledge:\n"
    "Commonsense Knowledge: Knowledge about the world that humans learn from their everyday experiences (e.g., many donuts being made in a cart implies they are for sale rather than for personal consumption).\n"
    "Visual Knowledge: Knowledge of concepts represented visually (e.g., muted color palettes are associated with the 1950s).\n"
    "Cultural Knowledge: Understanding cultural references, norms, and practices (e.g., knowing that a red envelope is associated with good luck in Chinese culture).\n"
    "Temporal Knowledge: Awareness of historical events, timelines, and changes over time (e.g., recognizing a specific style of clothing as being from the 1980s).\n"
    "Geographical Knowledge: Information about locations, landmarks, and regional characteristics (e.g., identifying a famous monument like the Eiffel Tower in Paris).\n"
    "Social Knowledge: Understanding social interactions, relationships, and behaviors (e.g., recognizing that a handshake is a form of greeting).\n"
    "Scientific Knowledge: Knowledge from various scientific domains like physics, biology, chemistry, astronomy, etc. (e.g., understanding that certain plants are poisonous).\n"
    "Technical Knowledge: Familiarity with technology, machinery, and tools (e.g., identifying parts of a computer or types of construction equipment).\n"
    "Mathematical Knowledge: Basic mathematical concepts and their applications (e.g., understanding geometric shapes or calculating areas).\n"
    "Literary Knowledge: Awareness of literature, authors, and genres (e.g., recognizing characters from classic novels)."
    "There is an image which can be described as: {caption}.\n"
    "The image has following objects: {object_tags}.\n"
    "A user is asking the following question on the image: {question}.\n"
    "What type of knowledge is required to answer the question? Choose one or many alternatives from the above options.\n"
    "Return output as list of strings as JSON Object. Example: {{'knowledge_type': ['knowledge_a', 'knowledge_b']}}"
)

domain_prompt = (
    "Following are source application domains:\n"
    "Anthropology, Books, Computer Science, Economics, Fiction, Formal logic, Government and Politics, History, "
    "Justice, Knowledge Base, Law, Linguistics, Movies, Mathematics, Nature, News, Nutrition and Food, Professions,"
    " Public Places, Reviews, Science, Social Media, Sports.\n"
    "There is an image which can be described as: {caption}.\n"
    "The image has the following objects: {object_tags}.\n"
    "A user is asking the following question on the image: {question},\n"
    "What type of application domain does this task belong to? Choose one or many alternatives from the above options.\n"
    "Return output as list of strings as JSON Object. Example: {{'application_domain': ['domain_a', 'domain_b']}}"
)

phi_3_vision_prompt_template = '<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n'
