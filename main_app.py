import const

import time
import streamlit as st
st.set_page_config(**const.SET_PAGE_CONFIG)
st.markdown(const.HIDE_ST_STYLE, unsafe_allow_html=True)
from PIL import Image
import numpy as np
import pandas as pd
import requests
from numpy.linalg import norm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, AutoTokenizer
from st_img_pastebutton import paste
from io import BytesIO
import base64



def main(): 
    ''''''
    st.markdown(f'<h1 style="text-align:center;">Image2Emoji App</h1>', unsafe_allow_html=True)

    # Load the model cached
    @st.cache_resource
    def load_model():
        processor = CLIPImageProcessor.from_pretrained(const.MODEL)
        tokenizer = AutoTokenizer.from_pretrained(const.MODEL)
        vision_model = CLIPVisionModelWithProjection.from_pretrained(const.MODEL)
        text_model = CLIPTextModelWithProjection.from_pretrained(const.MODEL)

        return processor, tokenizer, vision_model, text_model
    processor, tokenizer, vision_model, text_model = load_model()

    # Load the text embeddings
    @st.cache_resource
    def load_data():
        text_embeddings = np.load('EmojiDataset/embeddings.npy')
        df_emoji = pd.read_csv('EmojiDataset/full_emoji.csv', usecols=['emoji', 'name'])
        text = df_emoji['name'].tolist()
        emoji = df_emoji['emoji'].tolist()
        return text_embeddings, text, emoji
    text_embeddings, text, emoji = load_data()
    ''''''
    
    #col1, col2 = st.columns([1, 1])
    img_source = st.radio('Image Source', ('Sample', 'Upload', 'Paste', 'None'), help='You can paste an mathematical formula image from clipboard or upload an image from your local machine.')
    if img_source == 'Sample':
        try:
            image_data = Image.open(requests.get(const.URL, stream=True).raw)
        except:
            image_data = None
    elif img_source == 'Paste':
        pasted_img = paste(key='image_clipboard', label='Paste an image from clipboard')
        try:
            header, encoded = pasted_img.split(",", 1)
            binary_data = base64.b64decode(encoded)
            image_data = Image.open(BytesIO(binary_data)).convert('RGB')
        except:
            image_data = None
    elif img_source == 'Upload':
        image_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
        try :
            image_data = Image.open(image_file)
        except:
            image_data = None
    else:
        image_data = None


    if image_data is not None:
        with st.spinner('Loading...'):
            start_time = time.time()
            st.image(image_data, caption='Uploaded image', use_container_width=True)
            inputs = processor(image_data, return_tensors='pt')
            outputs = vision_model(**inputs)
            image_embeddings = outputs.image_embeds.detach().cpu().numpy() 

            col1, col2 = st.columns([1, 5])
            col1.button('Reload', help='Reload all process.')
            col2.success(f'Elapsed time: {time.time()-start_time:.2f} [sec]')

    else:
        image_embeddings = np.zeros((1, 512))

    col1, col2, col3 = st.columns([1, 1, 1])
    pos_prompt = col1.expander('Positive Prompt').text_area(label='', help='Input a positive prompt for the image (optional).', height=50)
    neg_prompt = col2.expander('Negative Prompt').text_area(label='', help='Input a negative prompt for the image (optional).', height=50)

    #button('Reload', help='Reload the output of the model.')
    if pos_prompt is not None:
        inputs = tokenizer([pos_prompt], return_tensors='pt', padding=True)
        pos_prompt_embeddings = text_model(**inputs).text_embeds.detach().cpu().numpy()
    else:
        pos_prompt_embeddings = None
    
    if neg_prompt is not None:
        inputs = tokenizer([neg_prompt], return_tensors='pt', padding=True)
        neg_prompt_embeddings = text_model(**inputs).text_embeds.detach().cpu().numpy()
    else:
        neg_prompt_embeddings = None

    # Calculate the similarity
    ratio = col3.slider('Prompt Ratio', min_value=0.0, max_value=1.0, value=0.2, step=0.1, help='The ratio of positive prompt to negative prompt (optional).')
    image_embeddings = (1-ratio)*image_embeddings + ratio*(pos_prompt_embeddings - neg_prompt_embeddings)
    sim = np.dot(image_embeddings, text_embeddings.T) / (norm(image_embeddings) * norm(text_embeddings, axis=1))

    # Get the top 5 emojis
    top5_idx = np.argsort(sim[0])[::-1][:5]
    top5_emoji = [emoji[i] for i in top5_idx]
    top5_text = [text[i] for i in top5_idx]
    top5_sim = sim[0][top5_idx]
    
    # Display the result
    st.subheader('Top 5 Emojis')
    if not np.isnan(top5_sim[0]):
        for i in range(5):
            st.write(f'{top5_sim[i]:.3f} : {top5_emoji[i]} ({top5_text[i]})')
    
    # Footer
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align:center;">Image2Emoji App</h2>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;font-size:12px;opacity:0.7;">Source code is <a href="https://github.com/yosshstd/Image2Emoji" target="_blank">here</a></div>',
        unsafe_allow_html=True
    )
    st.markdown('<br>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
