import React, { useState, useEffect } from 'react';
import './App.css';
import ImageUploader from './ImageUploader';
import { Button } from '@mui/material';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Link from '@mui/material/Link';

function App() {
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    // Update body class when isDarkMode changes
    document.body.className = isDarkMode ? 'dark-mode' : '';
    document.getElementsByTagName('span')[0].style.color = isDarkMode ? 'lightblue' : 'blue';
    document.getElementsByTagName('span')[1].style.color = isDarkMode ? 'lightblue' : 'blue';
  }, [isDarkMode]);

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
  };

  return (
    <div className="App">
      <div style={{justifyContent: 'center', display: 'flex'}}>
      <Box sx={{ width: '80%', maxWidth: 700, position: 'relative'}}>
        <Typography variant="h2" gutterBottom>
          Scene OCR
        </Typography>
        <Typography variant="subtitle2" gutterBottom style={{ textAlign: 'left' }}>
        Demo of Scene-ocr from the repo: 
        <Link href="https://github.com/nirkoren1/Text_Detection" underline="hover">
          {' https://github.com/nirkoren1/Text_Detection'}
        </Link>
        <Typography variant="subtitle2" gutterBottom>
          click on a picture or upload one, then click on <span style={{ color: 'lightblue' }}>PREDICT</span><br></br> click on <span style={{ color: 'blue' }}>SHOW DETECTION</span> to see the heatmap image
        </Typography>
        </Typography>
        
      </Box>
      </div>
      <Button onClick={toggleDarkMode} style={{position: 'fixed', left: 10, top: 10}}>
        {isDarkMode ? <Brightness7Icon /> : <Brightness4Icon />}
      </Button>
      
      <ImageUploader />
    </div>
  );
}

export default App;
