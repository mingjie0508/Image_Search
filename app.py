import streamlit as st
from image_search import Search_Setup

def compute_index():
    if st.session_state['image_list']:
        with st.spinner('Indexing images...'):
            # Set up the search engine
            st.session_state['ss'] = Search_Setup(
                image_list=st.session_state['image_list'], 
                model_name='openai/clip-vit-base-patch16',
                batch_size=1
            )
            # compute embedding for each image
            st.session_state['ss'].run_index()

# title
st.title('Image Search Engine')
st.text("Search images by text")

# upload images to search for
st.subheader('Step 1: Select images to search from')
image_list = st.file_uploader(
    'Images: ', type=['jpg', 'png'], accept_multiple_files=True, 
    on_change=compute_index, key='image_list'
)

# search image
st.subheader('Step 2: Search for images by text')
if image_list:  # if images have been uploaded
    query = st.text_input('Search: ')
    number = st.number_input("Show top K images: ", min_value=1, max_value=len(image_list), value=1, step=1)
    is_search = st.button('Search')
else:           # if images have not been uploaded
    query = st.text_input('Search: ', disabled=True)
    number = st.number_input("Show top K images: ", min_value=1, max_value=1, value=1, step=1, disabled=True)
    is_search = st.button('Search', disabled=True)

# start search
if is_search:
    with st.spinner('Searching for images...'):
        output = st.session_state['ss'].get_images_by_text(text=query, number_of_images=number)

    # display search results
    num_columns = 2
    cols = st.columns(num_columns)
    for i, p in enumerate(output):
        with cols[i % num_columns]:
            st.image(
                p['image'], caption=f"Score: {p['score']:.1f}", use_container_width=True
            )