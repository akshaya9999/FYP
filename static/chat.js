// ========== State ==========
let currentUser = null;
let isSignupMode = false;

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
                toggleLink.click(); // Switch to sign in form
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
// ========== Voice to Text (Web Speech API) ==========

const voiceBtn = document.getElementById('voiceBtn');

let recognition;
if ('webkitSpeechRecognition' in window) {
    recognition = new webkitSpeechRecognition();
    recognition.lang = "en-US";
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = () => {
        voiceBtn.classList.add("listening");
        voiceBtn.textContent = "🎙️ Listening...";
    };

    recognition.onend = () => {
        voiceBtn.classList.remove("listening");
        voiceBtn.textContent = "🎤";
    };

    recognition.onresult = (event) => {
        const speechText = event.results[0][0].transcript;
        console.log("Voice Input:", speechText);

        // Fill into query box
        queryInput.value = speechText;

        // Auto-send the message
        sendMessage();
    };
}

voiceBtn.addEventListener("click", () => {
    if (recognition) recognition.start();
    else alert("Your browser does not support voice input.");
});

// ========== Chat Functions ==========
async function loadChatHistory() {
    try {
        const res = await fetch('/history');
        const data = await res.json();
        
        if (data.history && data.history.length > 0) {
            messagesDiv.innerHTML = '';
            data.history.forEach(msg => {
                appendMessage(msg.role, msg.message);
            });
        }
    } catch (err) {
        console.error('Failed to load history:', err);
    }
}

function appendMessage(role, text) {
    const emptyState = messagesDiv.querySelector('.empty-state');
    if (emptyState) emptyState.remove();
    
    const msgDiv = document.createElement('div');
    msgDiv.className = `msg ${role}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'msg-content';
    contentDiv.innerHTML = `<strong>${role === 'user' ? 'You' : 'Bot'}:</strong> ${escapeHtml(text)}`;
    
    msgDiv.appendChild(contentDiv);
    messagesDiv.appendChild(msgDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

async function sendMessage() {
    const query = queryInput.value.trim();
    if (!query) return;

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
            appendMessage('bot', data.response);
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




