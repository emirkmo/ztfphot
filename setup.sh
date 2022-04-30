mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"emir.k@phys.au.dk\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml