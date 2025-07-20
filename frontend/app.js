const { useState, useEffect, useRef } = React;

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Login Component
function Login({ onLogin }) {
    const [userId, setUserId] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleLogin = async (e) => {
        e.preventDefault();
        if (!userId.trim()) return;
        
        setIsLoading(true);
        // Simulate login process
        setTimeout(() => {
            onLogin(userId);
            setIsLoading(false);
        }, 500);
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-900 to-gray-800">
            <div className="bg-gray-800 p-8 rounded-lg shadow-2xl w-96 border border-gray-700">
                <div className="text-center mb-8">
                    <h1 className="text-3xl font-bold text-white mb-2">Abundance AI Suite</h1>
                    <p className="text-gray-300">Enter your User ID to continue</p>
                </div>
                
                <form onSubmit={handleLogin} className="space-y-6">
                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                            User ID
                        </label>
                        <input
                            type="text"
                            value={userId}
                            onChange={(e) => setUserId(e.target.value)}
                            placeholder="Enter your user ID"
                            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-white placeholder-gray-400"
                            required
                        />
                    </div>
                    
                    <button
                        type="submit"
                        disabled={isLoading || !userId.trim()}
                        className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                        {isLoading ? (
                            <span className="flex items-center justify-center">
                                <i className="fas fa-spinner spinner mr-2"></i>
                                Logging in...
                            </span>
                        ) : (
                            'Continue'
                        )}
                    </button>
                </form>
            </div>
        </div>
    );
}

// Status Indicator Component
function StatusIndicator({ status, message }) {
    if (!status) return null;

    const getStatusIcon = () => {
        switch (status) {
            case 'searching_internet':
                return 'fas fa-globe text-blue-400';
            case 'analyzing_documents':
                return 'fas fa-file-alt text-green-400';
            case 'thinking':
                return 'fas fa-brain text-purple-400';
            case 'complete':
                return 'fas fa-check text-green-400';
            case 'error':
                return 'fas fa-exclamation-triangle text-red-400';
            default:
                return 'fas fa-cog text-gray-400';
        }
    };

    const getStatusText = () => {
        switch (status) {
            case 'searching_internet':
                return 'Searching the internet...';
            case 'analyzing_documents':
                return 'Analyzing documents...';
            case 'thinking':
                return 'Generating response...';
            case 'complete':
                return 'Response complete';
            case 'error':
                return 'Error occurred';
            default:
                return 'Processing...';
        }
    };

    const getStatusColor = () => {
        switch (status) {
            case 'searching_internet':
                return 'bg-blue-900/50 border-blue-700 text-blue-300';
            case 'analyzing_documents':
                return 'bg-green-900/50 border-green-700 text-green-300';
            case 'thinking':
                return 'bg-purple-900/50 border-purple-700 text-purple-300';
            case 'complete':
                return 'bg-green-900/50 border-green-700 text-green-300';
            case 'error':
                return 'bg-red-900/50 border-red-700 text-red-300';
            default:
                return 'bg-gray-800 border-gray-600 text-gray-300';
        }
    };

    return (
        <div className={`flex items-center space-x-2 text-sm px-3 py-2 rounded-lg border ${getStatusColor()}`}>
            <i className={`fas fa-spinner spinner ${getStatusIcon().split(' ')[1]}`}></i>
            <span>{message || getStatusText()}</span>
        </div>
    );
}

// File Upload Component
function FileUpload({ onFilesSelected, isUploading }) {
    const [dragActive, setDragActive] = useState(false);
    const fileInputRef = useRef(null);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            onFilesSelected(Array.from(e.dataTransfer.files));
        }
    };

    const handleFileSelect = (e) => {
        if (e.target.files) {
            onFilesSelected(Array.from(e.target.files));
        }
    };

    return (
        <div className="mb-4">
            <div
                className={`file-drop-zone rounded-lg p-4 text-center cursor-pointer ${dragActive ? 'dragover' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept=".pdf,.docx,.doc,.txt,.md,.pptx,.ppt,.xlsx,.xls"
                    onChange={handleFileSelect}
                    className="hidden"
                />
                
                {isUploading ? (
                    <div className="flex items-center justify-center space-x-2">
                        <i className="fas fa-spinner spinner text-blue-400"></i>
                        <span className="text-blue-300">Uploading documents...</span>
                    </div>
                ) : (
                    <div>
                        <i className="fas fa-cloud-upload-alt text-4xl text-gray-500 mb-2"></i>
                        <p className="text-gray-300">
                            Drop files here or click to upload
                        </p>
                        <p className="text-sm text-gray-400 mt-1">
                            Supports PDF, DOCX, TXT, MD, PPTX, XLSX
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}

// Chat Message Component
function ChatMessage({ message, isUser, isStreaming = false }) {
    if (isUser) {
        return (
            <div className="flex justify-end">
                <div className="bg-blue-600 text-white px-4 py-2 rounded-lg max-w-xs lg:max-w-md">
                    <p className="text-sm">{message}</p>
                </div>
            </div>
        );
    }

    if (message.includes('📎 Uploaded')) {
        return (
            <div className="flex justify-start">
                <div className="bg-gray-700 text-gray-300 px-4 py-2 rounded-lg max-w-xs lg:max-w-md border border-gray-600">
                    <p className="text-sm">{message}</p>
                </div>
            </div>
        );
    }

    return (
        <div className="flex justify-start">
            <div className="bg-gray-800 text-gray-100 px-4 py-2 rounded-lg max-w-xs lg:max-w-md border border-gray-700">
                <p className={`text-sm ${isStreaming ? 'streaming-text' : ''}`}>
                    {message}
                    {isStreaming && (
                        <span className="inline-block w-2 h-4 bg-blue-400 ml-1 animate-pulse"></span>
                    )}
                </p>
            </div>
        </div>
    );
}

// Chat Interface Component
function ChatInterface({ userId, onLogout }) {
    const [messages, setMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [isUploading, setIsUploading] = useState(false);
    const [status, setStatus] = useState(null);
    const [statusMessage, setStatusMessage] = useState('');
    const [showSidePanel, setShowSidePanel] = useState(false);
    const [chatHistory, setChatHistory] = useState([]);
    const [showDeleteMenu, setShowDeleteMenu] = useState(false);
    const [showDeleteChatDialog, setShowDeleteChatDialog] = useState(false);
    const [showDeleteUserDialog, setShowDeleteUserDialog] = useState(false);
    const [isDeleting, setIsDeleting] = useState(false);
    const [notifications, setNotifications] = useState([]);
    const messagesEndRef = useRef(null);
    const sidePanelRef = useRef(null);
    const deleteMenuRef = useRef(null);
    
    // Initialize thread_id from localStorage or generate new one
    const threadId = useRef(() => {
        const savedThreadId = localStorage.getItem(`thread_id_${userId}`);
        if (savedThreadId) {
            return savedThreadId;
        }
        const newThreadId = `${userId}_${crypto.randomUUID()}`;
        localStorage.setItem(`thread_id_${userId}`, newThreadId);
        return newThreadId;
    })();

    const showNotification = (message, type = 'success') => {
        setNotifications(prev => [...prev, { message, type }]);
        setTimeout(() => setNotifications(prev => prev.filter(n => n.message !== message)), 5000); // Auto-hide after 5 seconds
    };

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Close dropdown when clicking outside
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (showDeleteMenu && !event.target.closest('.user-menu')) {
                setShowDeleteMenu(false);
            }
        };

        const handleEscape = (event) => {
            if (event.key === 'Escape' && showDeleteMenu) {
                setShowDeleteMenu(false);
            }
        };

        const handleKeyboardShortcuts = (event) => {
            // Ctrl+N for new chat
            if (event.ctrlKey && event.key === 'n') {
                event.preventDefault();
                handleNewChat();
            }
            // Ctrl+B for side panel
            if (event.ctrlKey && event.key === 'b') {
                event.preventDefault();
                toggleSidePanel();
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        document.addEventListener('keydown', handleEscape);
        document.addEventListener('keydown', handleKeyboardShortcuts);
        
        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
            document.removeEventListener('keydown', handleEscape);
            document.removeEventListener('keydown', handleKeyboardShortcuts);
        };
    }, [showDeleteMenu]);

    const handleLogout = () => {
        if (onLogout) {
            onLogout();
        }
    };

    const handleNewChat = () => {
        setMessages([]);
        setUploadedFiles([]);
        // Generate unique thread_id using UUID to prevent collisions
        const newThreadId = `${userId}_${crypto.randomUUID()}`;
        threadId.current = newThreadId;
        // Save to localStorage
        localStorage.setItem(`thread_id_${userId}`, newThreadId);
        showNotification('New chat started!', 'success');
    };

    const handleThreadSelect = (selectedThreadId) => {
        threadId.current = selectedThreadId;
        setMessages([]);
        setUploadedFiles([]);
        setShowSidePanel(false);
        // Save to localStorage
        localStorage.setItem(`thread_id_${userId}`, selectedThreadId);
        showNotification(`Switched to chat: ${selectedThreadId}`, 'success');
    };

    const toggleSidePanel = () => {
        setShowSidePanel(!showSidePanel);
    };

    const handleDeleteChat = async () => {
        setShowDeleteChatDialog(true);
        setShowDeleteMenu(false);
    };

    const confirmDeleteChat = async () => {
        setIsDeleting(true);
        try {
            const response = await fetch(`${API_BASE_URL}/delete_thread`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: userId,
                    thread_id: threadId.current
                })
            });

            if (response.ok) {
                // Clear messages and create new thread with UUID
                setMessages([]);
                setUploadedFiles([]);
                const newThreadId = `${userId}_${crypto.randomUUID()}`;
                threadId.current = newThreadId;
                // Save to localStorage
                localStorage.setItem(`thread_id_${userId}`, newThreadId);
                showNotification('Chat deleted successfully!', 'success');
            } else {
                const error = await response.json();
                showNotification(`Error deleting chat: ${error.detail || 'Unknown error'}`, 'error');
            }
        } catch (error) {
            console.error('Error deleting chat:', error);
            showNotification('Error deleting chat. Please try again.', 'error');
        } finally {
            setIsDeleting(false);
            setShowDeleteChatDialog(false);
        }
    };

    const handleDeleteUser = async () => {
        setShowDeleteUserDialog(true);
        setShowDeleteMenu(false);
    };

    const confirmDeleteUser = async () => {
        setIsDeleting(true);
        try {
            const response = await fetch(`${API_BASE_URL}/delete_user_document`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: userId
                })
            });

            if (response.ok) {
                showNotification('Account deleted successfully!', 'success');
                handleLogout();
            } else {
                const error = await response.json();
                showNotification(`Error deleting account: ${error.detail || 'Unknown error'}`, 'error');
            }
        } catch (error) {
            console.error('Error deleting account:', error);
            showNotification('Error deleting account. Please try again.', 'error');
        } finally {
            setIsDeleting(false);
            setShowDeleteUserDialog(false);
        }
    };

    const handleFileUpload = async (files) => {
        setIsUploading(true);
        
        try {
            // Create FormData to send files to backend
            const formData = new FormData();
            formData.append('user_id', userId);
            formData.append('thread_id', threadId.current);
            
            // Add each file to FormData
            Array.from(files).forEach(file => {
                formData.append('files', file);
            });

            // Send files to backend for indexing
            const response = await fetch(`${API_BASE_URL}/upload_and_index`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Failed to upload files');
            }

            const result = await response.json();
            
            // Add system message about uploaded files
            setMessages(prev => [...prev, {
                id: Date.now(),
                text: `📎 Uploaded and indexed ${files.length} document(s): ${files.map(f => f.name).join(', ')} (${result.indexed_count} chunks indexed)`,
                isUser: false,
                isSystem: true
            }]);

            showNotification(`Successfully indexed ${result.indexed_count} document chunks`, 'success');

        } catch (error) {
            console.error('Error uploading files:', error);
            showNotification('Error uploading files. Please try again.', 'error');
        } finally {
            setIsUploading(false);
        }
    };

    const simulateStreamingResponse = async (responseText, statusUpdates = []) => {
        const words = responseText.split(' ');
        let currentText = '';
        
        for (let i = 0; i < words.length; i++) {
            // Update status if provided
            if (statusUpdates[i]) {
                setStatus(statusUpdates[i].status);
                setStatusMessage(statusUpdates[i].message);
            }
            
            currentText += (i > 0 ? ' ' : '') + words[i];
            
            setMessages(prev => {
                const newMessages = [...prev];
                const lastMessage = newMessages[newMessages.length - 1];
                if (lastMessage && !lastMessage.isUser) {
                    lastMessage.text = currentText;
                    lastMessage.isStreaming = i < words.length - 1;
                }
                return newMessages;
            });
            
            // Simulate typing delay
            await new Promise(resolve => setTimeout(resolve, 50 + Math.random() * 100));
        }
        
        // Clear status when done
        setStatus(null);
        setStatusMessage('');
    };

    const sendMessage = async () => {
        if (!inputMessage.trim() || isLoading) return;

        const messageText = inputMessage; // Store the message before clearing

        const userMessage = {
            id: Date.now(),
            text: messageText,
            isUser: true
        };

        setMessages(prev => [...prev, userMessage]);
        setInputMessage('');
        setIsLoading(true);

        // Add bot message placeholder
        const botMessageId = Date.now() + 1;
        setMessages(prev => [...prev, {
            id: botMessageId,
            text: '',
            isUser: false,
            isStreaming: true
        }]);

        try {
            // Use streaming endpoint
            const response = await fetch(`${API_BASE_URL}/chat/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: userId,
                    thread_id: threadId.current,
                    query: messageText, // Use stored message
                    use_attachment: false, // Documents are handled by file upload
                    use_search: false // Internet search is handled by file upload
                })
            });

            if (!response.ok) {
                throw new Error('Failed to get response');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let currentText = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            
                            if (data.type === 'status') {
                                setStatus(data.status);
                                setStatusMessage(data.message);
                                console.log('Status update:', data.status, data.message);
                            } else if (data.type === 'content') {
                                currentText = data.content;
                                setMessages(prev => {
                                    const newMessages = [...prev];
                                    const lastMessage = newMessages[newMessages.length - 1];
                                    if (lastMessage && !lastMessage.isUser) {
                                        lastMessage.text = currentText;
                                        lastMessage.isStreaming = !data.is_complete;
                                    }
                                    return newMessages;
                                });
                                
                                // Only clear status when content is complete
                                if (data.is_complete) {
                                    setTimeout(() => {
                                        setStatus(null);
                                        setStatusMessage('');
                                    }, 1000); // Keep status visible for 1 second after completion
                                }
                            }
                        } catch (e) {
                            console.error('Error parsing SSE data:', e);
                        }
                    }
                }
            }

            // Clear status when done
            setStatus(null);
            setStatusMessage('');

        } catch (error) {
            console.error('Error sending message:', error);
            setMessages(prev => {
                const newMessages = [...prev];
                const lastMessage = newMessages[newMessages.length - 1];
                if (lastMessage && !lastMessage.isUser) {
                    lastMessage.text = 'Sorry, I encountered an error. Please try again.';
                    lastMessage.isStreaming = false;
                }
                return newMessages;
            });
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    return (
        <div className="flex flex-col h-screen bg-gray-900">
            {/* Header */}
            <div className="bg-gray-800 border-b border-gray-700 px-6 py-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                        <button
                            onClick={toggleSidePanel}
                            className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
                        >
                            <i className="fas fa-bars"></i>
                        </button>
                        <h1 className="text-xl font-semibold text-white">Abundance AI Suite</h1>
                        <span className="text-sm text-gray-300">User: {userId}</span>
                        <span className="text-xs text-gray-400">(Ctrl+N: New Chat, Ctrl+B: History)</span>
                    </div>
                    <div className="flex items-center space-x-2">
                        <StatusIndicator status={status} message={statusMessage} />
                        
                        {/* User Menu */}
                        <div className="relative user-menu">
                            <button
                                onClick={() => setShowDeleteMenu(!showDeleteMenu)}
                                className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
                                disabled={isDeleting}
                            >
                                <i className="fas fa-ellipsis-v"></i>
                            </button>
                            
                            {showDeleteMenu && (
                                <div className="absolute right-0 mt-2 w-48 bg-gray-800 border border-gray-600 rounded-lg shadow-lg z-50">
                                    <div className="py-1">
                                        <button
                                            onClick={handleLogout}
                                            className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 flex items-center space-x-2"
                                        >
                                            <i className="fas fa-sign-out-alt text-gray-400"></i>
                                            <span>Logout</span>
                                        </button>
                                        
                                        <button
                                            onClick={handleNewChat}
                                            className="w-full text-left px-4 py-2 text-sm text-blue-400 hover:bg-blue-900/50 flex items-center space-x-2"
                                        >
                                            <i className="fas fa-plus-circle text-blue-400"></i>
                                            <span>New Chat</span>
                                        </button>

                                        <button
                                            onClick={handleDeleteChat}
                                            className="w-full text-left px-4 py-2 text-sm text-orange-400 hover:bg-orange-900/50 flex items-center space-x-2"
                                            disabled={isDeleting}
                                        >
                                            <i className="fas fa-trash text-orange-400"></i>
                                            <span>{isDeleting ? 'Deleting...' : 'Delete Chat'}</span>
                                        </button>
                                        
                                        <button
                                            onClick={handleDeleteUser}
                                            className="w-full text-left px-4 py-2 text-sm text-red-400 hover:bg-red-900/50 flex items-center space-x-2"
                                            disabled={isDeleting}
                                        >
                                            <i className="fas fa-user-times text-red-400"></i>
                                            <span>{isDeleting ? 'Deleting...' : 'Delete Account'}</span>
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* File Upload */}
            {uploadedFiles.length === 0 && (
                <FileUpload onFilesSelected={handleFileUpload} isUploading={isUploading} />
            )}

            {/* Uploaded Files Display */}
            {uploadedFiles.length > 0 && (
                <div className="px-6 py-2 bg-blue-900/30 border-b border-blue-700/50">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                            <i className="fas fa-paperclip text-blue-400"></i>
                            <span className="text-sm text-blue-300">
                                {uploadedFiles.length} document(s) attached
                            </span>
                        </div>
                        <button
                            onClick={() => setUploadedFiles([])}
                            className="text-sm text-blue-400 hover:text-blue-300"
                        >
                            Clear all
                        </button>
                    </div>
                </div>
            )}

            {/* Messages */}
            <div className="flex-1 overflow-y-auto px-6 py-4 message-container bg-gray-900">
                <div className="space-y-4">
                    {messages.map((message) => (
                        <ChatMessage
                            key={message.id}
                            message={message.text}
                            isUser={message.isUser}
                            isStreaming={message.isStreaming}
                        />
                    ))}
                </div>
                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="bg-gray-800 border-t border-gray-700 px-6 py-4">
                {/* Toggle Controls */}
                <div className="flex items-center space-x-4 mb-3">
                    <label className="flex items-center space-x-2 cursor-pointer">
                        <input
                            type="checkbox"
                            checked={false} // Documents are handled by file upload
                            onChange={(e) => {}}
                            className="rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-blue-500 focus:ring-offset-gray-800"
                        />
                        <span className="text-sm text-gray-300">Document Search</span>
                    </label>
                    
                    <label className="flex items-center space-x-2 cursor-pointer">
                        <input
                            type="checkbox"
                            checked={false} // Internet search is handled by file upload
                            onChange={(e) => {}}
                            className="rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-blue-500 focus:ring-offset-gray-800"
                        />
                        <span className="text-sm text-gray-300">Internet Search</span>
                    </label>
                </div>

                {/* Message Input */}
                <div className="flex items-end space-x-3">
                    <div className="flex-1">
                        <textarea
                            value={inputMessage}
                            onChange={(e) => setInputMessage(e.target.value)}
                            onKeyPress={handleKeyPress}
                            placeholder="Type your message..."
                            className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-white placeholder-gray-400"
                            rows="1"
                            disabled={isLoading}
                        />
                    </div>
                    
                    <button
                        onClick={sendMessage}
                        disabled={!inputMessage.trim() || isLoading}
                        className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                        {isLoading ? (
                            <i className="fas fa-spinner spinner"></i>
                        ) : (
                            <i className="fas fa-paper-plane"></i>
                        )}
                    </button>
                </div>
            </div>

            {/* Confirmation Dialogs */}
            <ConfirmationDialog
                isOpen={showDeleteChatDialog}
                title="Delete Chat"
                message="Are you sure you want to delete this chat? This action cannot be undone."
                confirmText="Delete Chat"
                cancelText="Cancel"
                onConfirm={confirmDeleteChat}
                onCancel={() => setShowDeleteChatDialog(false)}
                type="warning"
            />

            <ConfirmationDialog
                isOpen={showDeleteUserDialog}
                title="Delete Account"
                message="Are you sure you want to delete your account? This will delete ALL your data including all chats and documents. This action cannot be undone."
                confirmText="Delete Account"
                cancelText="Cancel"
                onConfirm={confirmDeleteUser}
                onCancel={() => setShowDeleteUserDialog(false)}
                type="danger"
            />

            {/* Side Panel */}
            <SidePanel
                userId={userId}
                onThreadSelect={handleThreadSelect}
                currentThreadId={threadId.current}
                isOpen={showSidePanel}
                onToggle={toggleSidePanel}
            />

            {/* Overlay for side panel */}
            {showSidePanel && (
                <div 
                    className="fixed inset-0 bg-black bg-opacity-50 z-30"
                    onClick={toggleSidePanel}
                />
            )}

            {/* Notification Component */}
            {notifications.map((notification, index) => (
                <Notification
                    key={index}
                    message={notification.message}
                    type={notification.type}
                    onClose={() => setNotifications(prev => prev.filter((_, i) => i !== index))}
                />
            ))}
        </div>
    );
}

// Side Panel Component
function SidePanel({ userId, onThreadSelect, currentThreadId, isOpen, onToggle }) {
    const [chatHistory, setChatHistory] = useState([]);
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        if (isOpen && userId) {
            loadChatHistory();
        }
    }, [isOpen, userId]);

    const loadChatHistory = async () => {
        setIsLoading(true);
        try {
            const response = await fetch(`${API_BASE_URL}/chat_history/${userId}`);
            if (response.ok) {
                const data = await response.json();
                setChatHistory(data.threads || []);
            }
        } catch (error) {
            console.error('Error loading chat history:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const handleDeleteThread = async (threadId, threadTitle) => {
        if (!confirm(`Delete chat "${threadTitle}"? This action cannot be undone.`)) {
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/delete_thread`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: userId,
                    thread_id: threadId
                })
            });

            if (response.ok) {
                // Remove from local state
                setChatHistory(prev => prev.filter(thread => thread.thread_id !== threadId));
                alert('Chat deleted successfully!');
            } else {
                const error = await response.json();
                alert(`Error deleting chat: ${error.detail || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Error deleting chat:', error);
            alert('Error deleting chat. Please try again.');
        }
    };

    return (
        <div className={`fixed inset-y-0 left-0 z-40 w-80 bg-gray-800 border-r border-gray-700 transform transition-transform duration-300 ease-in-out ${isOpen ? 'translate-x-0' : '-translate-x-full'}`}>
            <div className="flex flex-col h-full">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-gray-700">
                    <h2 className="text-lg font-semibold text-white">Chat History</h2>
                    <button
                        onClick={onToggle}
                        className="text-gray-400 hover:text-white"
                    >
                        <i className="fas fa-times"></i>
                    </button>
                </div>

                {/* Chat List */}
                <div className="flex-1 overflow-y-auto p-4">
                    {isLoading ? (
                        <div className="flex items-center justify-center py-8">
                            <i className="fas fa-spinner spinner text-blue-400"></i>
                            <span className="ml-2 text-gray-300">Loading...</span>
                        </div>
                    ) : chatHistory.length === 0 ? (
                        <div className="text-center py-8">
                            <i className="fas fa-comments text-4xl text-gray-500 mb-2"></i>
                            <p className="text-gray-400">No previous chats</p>
                        </div>
                    ) : (
                        <div className="space-y-2">
                            {chatHistory.map((thread) => (
                                <div
                                    key={thread.thread_id}
                                    className={`w-full p-3 rounded-lg transition-colors ${
                                        currentThreadId === thread.thread_id
                                            ? 'bg-blue-600 text-white'
                                            : 'text-gray-300 hover:bg-gray-700'
                                    }`}
                                >
                                    <div className="flex items-center justify-between">
                                        <button
                                            onClick={() => onThreadSelect(thread.thread_id)}
                                            className="flex-1 text-left min-w-0"
                                        >
                                            <div className="flex items-center justify-between">
                                                <div className="flex-1 min-w-0">
                                                    <h3 className="font-medium truncate">{thread.title}</h3>
                                                    <p className="text-sm opacity-75 truncate">{thread.last_message}</p>
                                                </div>
                                                <div className="text-xs opacity-60 ml-2">
                                                    {thread.message_count} msgs
                                                </div>
                                            </div>
                                        </button>
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handleDeleteThread(thread.thread_id, thread.title);
                                            }}
                                            className="ml-2 p-1 text-red-400 hover:text-red-300 hover:bg-red-900/30 rounded"
                                            title="Delete chat"
                                        >
                                            <i className="fas fa-trash text-xs"></i>
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-gray-700">
                    <div className="text-xs text-gray-400 text-center">
                        Click on a chat to continue the conversation
                    </div>
                </div>
            </div>
        </div>
    );
}

// Notification Component
function Notification({ message, type = 'success', onClose }) {
    const getTypeStyles = () => {
        switch (type) {
            case 'success':
                return 'bg-green-900/80 border-green-600 text-green-200';
            case 'error':
                return 'bg-red-900/80 border-red-600 text-red-200';
            case 'warning':
                return 'bg-yellow-900/80 border-yellow-600 text-yellow-200';
            default:
                return 'bg-blue-900/80 border-blue-600 text-blue-200';
        }
    };

    const getIcon = () => {
        switch (type) {
            case 'success':
                return 'fas fa-check-circle text-green-400';
            case 'error':
                return 'fas fa-exclamation-circle text-red-400';
            case 'warning':
                return 'fas fa-exclamation-triangle text-yellow-400';
            default:
                return 'fas fa-info-circle text-blue-400';
        }
    };

    return (
        <div className={`fixed top-4 right-4 p-4 border rounded-lg shadow-lg z-50 ${getTypeStyles()}`}>
            <div className="flex items-center space-x-3">
                <i className={getIcon()}></i>
                <span className="text-sm font-medium">{message}</span>
                <button
                    onClick={onClose}
                    className="text-gray-400 hover:text-gray-200"
                >
                    <i className="fas fa-times"></i>
                </button>
            </div>
        </div>
    );
}

// Confirmation Dialog Component
function ConfirmationDialog({ isOpen, title, message, confirmText, cancelText, onConfirm, onCancel, type = 'danger' }) {
    if (!isOpen) return null;

    const getTypeStyles = () => {
        switch (type) {
            case 'danger':
                return 'bg-red-600 hover:bg-red-700';
            case 'warning':
                return 'bg-orange-600 hover:bg-orange-700';
            default:
                return 'bg-blue-600 hover:bg-blue-700';
        }
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4 border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
                <p className="text-gray-300 mb-6">{message}</p>
                <div className="flex space-x-3 justify-end">
                    <button
                        onClick={onCancel}
                        className="px-4 py-2 text-gray-300 bg-gray-700 rounded-lg hover:bg-gray-600 transition-colors"
                    >
                        {cancelText || 'Cancel'}
                    </button>
                    <button
                        onClick={onConfirm}
                        className={`px-4 py-2 text-white rounded-lg transition-colors ${getTypeStyles()}`}
                    >
                        {confirmText || 'Confirm'}
                    </button>
                </div>
            </div>
        </div>
    );
}

// Main App Component
function App() {
    const [userId, setUserId] = useState(null);

    const handleLogin = (userId) => {
        setUserId(userId);
    };

    const handleLogout = () => {
        setUserId(null);
    };

    if (!userId) {
        return <Login onLogin={handleLogin} />;
    }

    return <ChatInterface userId={userId} onLogout={handleLogout} />;
}

// Render the app
try {
    console.log('Attempting to render React app...');
    ReactDOM.render(<App />, document.getElementById('app'));
    console.log('React app rendered successfully!');
} catch (error) {
    console.error('Error rendering React app:', error);
    const errorDiv = document.getElementById('app');
    errorDiv.innerHTML = '<div class="p-4 text-center"><h1 class="text-2xl font-bold text-red-600 mb-4">Error Loading Application</h1><div class="text-gray-600 mb-4">There was an error loading the application.</div><div class="bg-red-100 p-4 rounded text-left text-sm"><strong>Error:</strong> ' + error.message + '</div><button onclick="location.reload()" class="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">Reload Page</button></div>';
}