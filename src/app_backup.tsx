import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

type ChatMessage = {
  role: 'user' | 'bot';
  content: string;
  fileUrl?: string; // 파일 URL 추가
};

type ChatSession = {
  id: string;
  title: string;
  messages: ChatMessage[];
};

const App: React.FC = () => {
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [inputMessage, setInputMessage] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  useEffect(() => {
    const storedSessions = sessionStorage.getItem('chatSessions');
    if (storedSessions) {
      const parsedSessions = JSON.parse(storedSessions);
      setChatSessions(parsedSessions);
      if (parsedSessions.length > 0) {
        setCurrentSessionId(parsedSessions[0].id);
      } else {
        createNewChat();
      }
    } else {
      createNewChat();
    }
  }, []);

  useEffect(() => {
    sessionStorage.setItem('chatSessions', JSON.stringify(chatSessions));
  }, [chatSessions]);

  const createNewChat = () => {
    const newSession: ChatSession = {
      id: Date.now().toString(),
      title: '새 채팅',
      messages: [],
    };
    setChatSessions([newSession, ...chatSessions]);
    setCurrentSessionId(newSession.id);
  };

  const handleSendMessage = async (messageContent: string) => {
    if (!messageContent.trim() && !selectedFile) return;

    if (!currentSessionId) {
      createNewChat();
    }

    let fileUrl = null;
    if (selectedFile) {
      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const response = await axios.post('http://localhost:8000/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        fileUrl = response.data.fileUrl;
      } catch (error) {
        console.error('Error uploading file:', error);
        return;
      }
    }

    const newMessage: ChatMessage = { role: 'user', content: messageContent, fileUrl };
    const updatedSessions = chatSessions.map(session => 
      session.id === currentSessionId 
        ? { 
            ...session, 
            messages: [...session.messages, newMessage],
            title: session.messages.length === 0 ? messageContent : session.title
          }
        : session
    );

    setChatSessions(updatedSessions);
    setInputMessage('');
    setSelectedFile(null);

    try {
      const response = await axios.post('http://localhost:8000/chat', {
        message: messageContent,
      });

      const botMessage: ChatMessage = { role: 'bot', content: response.data.message };
      const finalSessions = updatedSessions.map(session => 
        session.id === currentSessionId 
          ? { ...session, messages: [...session.messages, botMessage] }
          : session
      );

      setChatSessions(finalSessions);
    } catch (error) {
      console.error('Error sending message:', error);
      if (axios.isAxiosError(error)) {
        console.error('Axios error:', error.response?.data);
      }
    }
  };

  const deleteChat = (id: string) => {
    setChatSessions(prevSessions => prevSessions.filter(session => session.id !== id));
    if (currentSessionId === id) {
      setCurrentSessionId(null);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file);
    } else {
      alert('동영상 파일만 업로드할 수 있습니다.');
    }
  };

  const handleCopy = (content: string) => {
    navigator.clipboard.writeText(content).then(() => {
      alert('복사되었습니다.');
    });
  };

  const handleRetry = async (message: ChatMessage) => {
    if (!currentSessionId) return;

    try {
      const response = await axios.post('http://localhost:8000/chat', {
        message: message.content,
      });

      const botMessage: ChatMessage = { role: 'bot', content: response.data.message };
      const updatedSessions = chatSessions.map(session => 
        session.id === currentSessionId 
          ? { ...session, messages: [...session.messages, botMessage] }
          : session
      );

      setChatSessions(updatedSessions);
    } catch (error) {
      console.error('Error sending message:', error);
      if (axios.isAxiosError(error)) {
        console.error('Axios error:', error.response?.data);
      }
    }
  };

  const currentSession = chatSessions.find(session => session.id === currentSessionId);

  return (
    <div className="app-container">
      <div className="sidebar">
        <button onClick={createNewChat}>새 채팅</button>
        <div className="chat-list">
          {chatSessions.map(session => (
            <div 
              key={session.id} 
              className={`chat-item ${session.id === currentSessionId ? 'active' : ''}`}
            >
              <span onClick={() => setCurrentSessionId(session.id)}>
                {session.title.length > 30 ? session.title.substring(0, 30) + '...' : session.title}
              </span>
              <button 
                className="delete-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  deleteChat(session.id);
                }}
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="3 6 5 6 21 6"></polyline>
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                </svg>
              </button>
            </div>
          ))}
        </div>
      </div>
      <div className="chat-container">
        <div className="chat-messages">
          {currentSession?.messages.map((message, index) => (
            <div key={index} className={`message ${message.role}`}>
              {message.role === 'bot' && (
                <div className="bot-icon">
                  <img src="/chatgpt.png" alt="ChatGPT" />
                </div>
              )}
              <div className="message-content">
                {message.content}
                {message.fileUrl && (
                  <div className="file-preview">
                    <a href={message.fileUrl} target="_blank" rel="noopener noreferrer">파일 보기</a>
                  </div>
                )}
                {message.role === 'bot' && (
                  <div className="message-actions">
                    <button onClick={() => handleCopy(message.content)} title="복사">
                      <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                      </svg>
                    </button>
                    <button onClick={() => handleRetry(message)} title="다시 답변 요청">
                      <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.3"></path>
                      </svg>
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
        <div className="chat-input-container">
          <div className="chat-input">
            <input
              type="file"
              id="file-upload"
              style={{ display: 'none' }}
              accept="video/*"
              onChange={handleFileChange}
            />
            <label htmlFor="file-upload" className="file-upload-label" title="파일 첨부">
              📎
            </label>
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage(inputMessage)}
              placeholder="메시지를 입력하세요..."
            />
            <button onClick={() => handleSendMessage(inputMessage)} disabled={!inputMessage.trim() && !selectedFile}>전송</button>
          </div>
          <div className="creator-info">kt ds DX사업본부 Delivery COE담당 DX개발팀 김재현</div>
        </div>
      </div>
    </div>
  );
};

export default App;
