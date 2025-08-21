# Hand Gesture-Controlled Music Player

This is a web application that reimagines how you interact with music. [cite_start]Using your webcam, you can control playback with simple hand gestures like play, pause, and skip tracks[cite: 21, 23]. [cite_start]The project is built on Django and features a full user authentication system, personal music libraries, and a sleek, modern UI with a neon theme[cite: 22, 59].

---

## Key Features ✨

* [cite_start]**Gesture-Based Controls**: Manage your music playback hands-free through your webcam[cite: 23].
* [cite_start]**User Authentication**: A complete system for user signup, login, and logout to keep your playlists private[cite: 56].
* [cite_start]**Personalized Playlists**: Browse the main library and add your favorite songs to a personal "My Music" section[cite: 53].
* [cite_start]**Music Upload**: Easily upload new audio files to the shared music library[cite: 52].
* [cite_start]**Advanced Player**: Includes standard controls plus extra features like shuffle and repeat[cite: 24, 54].

---

## Technology Stack 🛠️

* [cite_start]**Backend**: Django [cite: 90]
* [cite_start]**Gesture Recognition**: MediaPipe, joblib [cite: 91]
* [cite_start]**Real-time Communication**: Django Channels, WebSockets [cite: 92]
* [cite_start]**Frontend**: HTML, CSS, JavaScript [cite: 93]
* [cite_start]**Database**: SQLite [cite: 96]

---

## Backend Breakdown: Django & Django Channels ⚙️

[cite_start]This project leverages the **Django** framework, using its Model-View-Template (MVT) architecture to structure the application[cite: 64].

* [cite_start]**Models (`models.py`)**: Two primary models define the database structure: `Song`, which stores track information and the audio file, and `MyMusic`, which links a user to a song, creating a personalized playlist[cite: 66, 100]. This ensures each user's music collection is kept separate.

* [cite_start]**Views (`views.py`)**: The views contain the core logic of the application[cite: 67]. [cite_start]They handle user authentication (login/signup), process song uploads, display the music library, and manage adding or removing tracks from a user's "My Music" list[cite: 103, 108]. [cite_start]Views are protected to ensure only logged-in users can access their personal playlists[cite: 87].

* **Templates**: The frontend is rendered using Django's templating engine. [cite_start]HTML files define the structure for the home page, player, library, and user login/signup forms[cite: 68].

[cite_start]For the real-time gesture control, **Django Channels** extends Django's capabilities to handle protocols beyond HTTP, like WebSockets[cite: 92]. [cite_start]When the MediaPipe script on the client-side detects a gesture, it sends a message through a WebSocket to Django Channels[cite: 70, 81]. [cite_start]The server processes this message in real-time and executes the corresponding player command (e.g., play, pause), which is then reflected on the web interface without needing to reload the page[cite: 82, 83]. This creates a fluid and interactive user experience.
