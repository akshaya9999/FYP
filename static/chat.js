// ========== State ==========
let currentUser = null;
let isSignupMode = false;
let lastUserQuery = ''; // Track last query for feedback attachment

// ========== DOM Elements ==========
const authContainer = document.getElementById('authContainer');
const chatbox = document.getElementById('chatbox');
const signinForm = document.getElementById('signinForm');
const signupForm = document.getElementById('signupForm');
const toggleLink = document.getElementById('toggleLink');
const toggleText = document.getElementById('toggleText');
const authTitle = document.getElementById('authTitle');
const errorMessage = document.getElementById('errorMessage');
const successMessage = document.getElementById('successMessage');
const userName = document.getElementById('userName');
const messagesDiv = document.getElementById('messages');
const queryInput = document.getElementById('query');
const fileInput = document.getElementById('fileInput');
const uploadForm = document.getElementById('uploadForm');

// ========== Utility Functions ==========
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    successMessage.style.display = 'none';
}

function showSuccess(message) {
    successMessage.textContent = message;
    successMessage.style.display = 'block';
    errorMessage.style.display = 'none';
}

function hideMessages() {
    errorMessage.style.display = 'none';
    successMessage.style.display = 'none';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ========== RLHF Feedback ==========
function renderFeedback(messageId, query, response, container) {
    // Don't add feedback bar if no messageId (e.g. history messages)
    if (!messageId) return;

    const bar = document.createElement('div');
    bar.className = 'feedback-bar';

    const thumbUp = document.createElement('button');
    thumbUp.className = 'feedback-btn';
    thumbUp.textContent = '👍';
    thumbUp.title = 'Good response';

    const thumbDown = document.createElement('button');
    thumbDown.className = 'feedback-btn';
    thumbDown.textContent = '👎';
    thumbDown.title = 'Bad response';

    const thanks = document.createElement('span');
    thanks.className = 'feedback-thanks';
    thanks.style.display = 'none';
    thanks.textContent = 'Thanks for your feedback!';

    bar.appendChild(thumbUp);
    bar.appendChild(thumbDown);
    bar.appendChild(thanks);
    container.appendChild(bar);

    async function submitFeedback(rating) {
        thumbUp.disabled = true;
        thumbDown.disabled = true;

        if (rating === 1) thumbUp.classList.add('selected-up');
        else thumbDown.classList.add('selected-down');

        thanks.style.display = 'inline';

        try {
            const res = await fetch('/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message_id: messageId,
                    query: query,
                    response: response,
                    rating: rating,
                }),
            });
            if (!res.ok) {
                thanks.textContent = 'Could not save feedback.';
            }
        } catch (err) {
            thanks.textContent = 'Could not save feedback.';
        }
    }

    thumbUp.addEventListener('click', () => submitFeedback(1));
    thumbDown.addEventListener('click', () => submitFeedback(0));
}

// ========== Auth Functions ==========
async function checkAuth() {
    try {
        const res = await fetch('/me');
        const data = await res.json();
        
        if (data.authenticated) {
            currentUser = data;
            showChat();
            loadChatHistory();
        } else {
            showAuth();
        }
    } catch (err) {
        console.error('Auth check failed:', err);
        showAuth();
    }
}

function showAuth() {
    authContainer.style.display = 'block';
    chatbox.style.display = 'none';
}

function showChat() {
    authContainer.style.display = 'none';
    chatbox.style.display = 'flex';
    userName.textContent = currentUser.name || currentUser.email;
}

toggleLink.addEventListener('click', () => {
    isSignupMode = !isSignupMode;
    hideMessages();
    
    if (isSignupMode) {
        signinForm.classList.add('hidden');
        signupForm.classList.remove('hidden');
        authTitle.textContent = 'Sign Up';
        toggleText.textContent = 'Already have an account?';
        toggleLink.textContent = 'Sign In';
    } else {
        signupForm.classList.add('hidden');
        signinForm.classList.remove('hidden');
        authTitle.textContent = 'Sign In';
        toggleText.textContent = "Don't have an account?";
        toggleLink.textContent = 'Sign Up';
    }
});

signinForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideMessages();
    
    const email = document.getElementById('signinEmail').value;
    const password = document.getElementById('signinPassword').value;
    
    try {
        const res = await fetch('/signin', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({email, password})
        });
        
        const data = await res.json();
        
        if (res.ok) {
            currentUser = data;
            showChat();
            loadChatHistory();
        } else {
            showError(data.error || 'Sign in failed');
        }
    } catch (err) {
        showError('Connection error. Please try again.');
    }
});

signupForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideMessages();
    
    const name = document.getElementById('signupName').value;
    const email = document.getElementById('signupEmail').value;
    const password = document.getElementById('signupPassword').value;
    
    try {
        const res = await fetch('/signup', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name, email, password})
        });
        
        const data = await res.json();
        
        if (res.ok) {
            showSuccess('Account created! Please sign in.');
            setTimeout(() => {
                toggleLink.click();
            }, 1500);
        } else {
            showError(data.error || 'Sign up failed');
        }
    } catch (err) {
        showError('Connection error. Please try again.');
    }
});

async function signOut() {
    try {
        await fetch('/signout', {method: 'POST'});
        currentUser = null;
        messagesDiv.innerHTML = '<div class="empty-state"><p>No messages yet. Upload a document and start chatting!</p></div>';
        showAuth();
    } catch (err) {
        console.error('Sign out error:', err);
    }
}

// ========== Voice Recording ==========
const voiceBtn = document.getElementById("voiceBtn");

let mediaRecorder;
let audioChunks = [];

voiceBtn.addEventListener("click", async () => {

    if (!mediaRecorder || mediaRecorder.state === "inactive") {

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();
        audioChunks = [];

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        voiceBtn.textContent = "⏹ Stop";

    } else {

        mediaRecorder.stop();
        voiceBtn.textContent = "🎤";

        mediaRecorder.onstop = async () => {

            const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
            const formData = new FormData();
            formData.append("audio", audioBlob, "voice.webm");

            const res = await fetch("/voice_chat", {
                method: "POST",
                body: formData
            });

            const data = await res.json();

            if (res.ok) {
                appendMessage("user", data.transcribed_text);
                // Pass message_id and transcribed text for feedback
                appendMessage("bot", data.response, data.message_id, data.transcribed_text);
            } else {
                appendMessage("bot", "❌ Voice processing failed.");
            }
        };
    }
});

// ========== Chat Functions ==========
async function loadChatHistory() {
    try {
        const res = await fetch('/history');
        const data = await res.json();
        
        if (data.history && data.history.length > 0) {
            messagesDiv.innerHTML = '';
            data.history.forEach(msg => {
                // History messages don't have message_id — no feedback bar
                appendMessage(msg.role, msg.message);
            });
        }
    } catch (err) {
        console.error('Failed to load history:', err);
    }
}

// Updated appendMessage — accepts optional messageId and query for feedback
function appendMessage(role, text, messageId = null, query = null) {
    const emptyState = messagesDiv.querySelector('.empty-state');
    if (emptyState) emptyState.remove();

    const msgDiv = document.createElement('div');
    msgDiv.className = `msg ${role}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'msg-content';
    contentDiv.innerHTML = `<strong>${role === 'user' ? 'You' : 'Bot'}:</strong> ${escapeHtml(text)}`;

    msgDiv.appendChild(contentDiv);

    // Add feedback bar below bot messages when messageId is present
    if (role === 'bot' && messageId) {
        const feedbackQuery = query || lastUserQuery;
        renderFeedback(messageId, feedbackQuery, text, msgDiv);
    }

    messagesDiv.appendChild(msgDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

async function sendMessage() {
    const query = queryInput.value.trim();
    if (!query) return;

    lastUserQuery = query; // Store for feedback reference
    appendMessage('user', query);
    queryInput.value = '';

    const typing = document.createElement('div');
    typing.className = 'msg bot';
    typing.innerHTML = `<div class='msg-content'><em>Typing...</em></div>`;
    messagesDiv.appendChild(typing);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query})
        });
        
        const data = await res.json();
        typing.remove();
        
        if (res.ok) {
            // Pass message_id and original query for feedback
            appendMessage('bot', data.response, data.message_id, query);
        } else {
            appendMessage('bot', `❌ Error: ${data.error || 'Unknown error'}`);
        }
    } catch (err) {
        typing.remove();
        appendMessage('bot', '❌ Error connecting to server.');
    }
}

// Handle Enter key
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// File upload
fileInput.addEventListener('change', (e) => {
    const fileName = e.target.files[0]?.name || 'Choose a file to upload...';
    document.getElementById('fileName').textContent = fileName;
});

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select a file first');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    const fileName = document.getElementById('fileName');
    fileName.textContent = 'Uploading...';
    
    try {
        const res = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await res.json();
        
        if (res.ok) {
            fileName.textContent = 'Upload successful! ✓';
            appendMessage('bot', `📄 ${data.message}`);
            setTimeout(() => {
                fileName.textContent = 'Choose a file to upload...';
                fileInput.value = '';
            }, 2000);
        } else {
            fileName.textContent = 'Upload failed ✗';
            appendMessage('bot', `❌ Upload error: ${data.error}`);
        }
    } catch (err) {
        fileName.textContent = 'Upload failed ✗';
        appendMessage('bot', '❌ Upload failed: Connection error');
    }
});

// ========== Initialize ==========
checkAuth();


