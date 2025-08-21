# Hand Gesture-Controlled Music Player

This is a web application that reimagines how you interact with music. Using your webcam, you can control playback with simple hand gestures like play, pause, and skip tracks. The project is built on Django and features a full user authentication system, personal music libraries, and a sleek, modern UI with a neon theme.

---

## Key Features ✨

* **Gesture-Based Controls**: Manage your music playback hands-free through your webcam.
* **User Authentication**: A complete system for user signup, login, and logout to keep your playlists private.
* **Personalized Playlists**: Browse the main library and add your favorite songs to a personal "My Music" section.
* **Music Upload**: Easily upload new audio files to the shared music library.
* **Advanced Player**: Includes standard controls plus extra features like shuffle and repeat.

---

## Technology Stack 🛠️

* **Backend**: Django 
* **Gesture Recognition**: MediaPipe, joblib 
* **Real-time Communication**: Django Channels, WebSockets 
* **Frontend**: HTML, CSS, JavaScript 
* **Database**: SQLite 

---

## Backend Breakdown: Django & Django Channels ⚙️

This project leverages the **Django** framework, using its Model-View-Template (MVT) architecture to structure the application.

* **Models (`models.py`)**: Two primary models define the database structure: `Song`, which stores track information and the audio file, and `MyMusic`, which links a user to a song, creating a personalized playlist. This ensures each user's music collection is kept separate.

* **Views (`views.py`)**: The views contain the core logic of the application. They handle user authentication (login/signup), process song uploads, display the music library, and manage adding or removing tracks from a user's "My Music" list. Views are protected to ensure only logged-in users can access their personal playlists.

* **Templates**: The frontend is rendered using Django's templating engine. HTML files define the structure for the home page, player, library, and user login/signup forms.

For the real-time gesture control, **Django Channels** extends Django's capabilities to handle protocols beyond HTTP, like WebSockets. When the MediaPipe script on the client-side detects a gesture, it sends a message through a WebSocket to Django Channels. The server processes this message in real-time and executes the corresponding player command (e.g., play, pause), which is then reflected on the web interface without needing to reload the page. This creates a fluid and interactive user experience.
