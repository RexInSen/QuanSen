
import sys, os
# Tell Streamlit where its static assets live inside the frozen bundle
os.environ.setdefault('STREAMLIT_STATIC_FOLDER',
    os.path.join(getattr(sys, '_MEIPASS', os.path.dirname(sys.executable)),
                 'streamlit', 'static'))
