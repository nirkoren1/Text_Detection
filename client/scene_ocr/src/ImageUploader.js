import React, { useState, useRef, useEffect } from 'react';
import CircularIndeterminate from './Loading'
import MultiActionAreaCard from './ImageCard'
import { Button } from '@mui/material';
import { styled } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import Tooltip from '@mui/material/Tooltip';
import StandardImageList from './ImageList';


function ImageUploader() {
  const [image, setImage] = useState(null);
  // const [imageString, setImageString] = useState('')
  const [previewUrl, setPreviewUrl] = useState('');
  const [loadingOrButton, setLoadingOrButton] = useState('')
  const inputElement = useRef(null)
  const mediaElement = useRef(null)
  const [bbs, setBbs] = useState([])
  const [imageWidth, setImageWidth] = useState(0);
  const [imageHeight, setImageHeight] = useState(0);
  const [windowSize, setWindowSize] = useState([
    window.innerWidth,
    window.innerHeight,
  ]);

  useEffect(() => {
    const handleWindowResize = () => {
      setWindowSize([window.innerWidth, window.innerHeight]);
    };

    window.addEventListener('resize', handleWindowResize);

    return () => {
      window.removeEventListener('resize', handleWindowResize);
    };
  }, []);

  let detectionImage = ''
  let listBB = null

  const changeToDetection = () => {
    setPreviewUrl(detectionImage)
    setLoadingOrButton(showImageButton)
    return
  }

  const changeToImage = () => {
    const img = inputElement.current.files[0]
    try {
      setPreviewUrl(URL.createObjectURL(image))
    } catch {
      setPreviewUrl(image)
    }
      

    if (detectionImage != '')
      setLoadingOrButton(detectionButton)
    return

  }

  useEffect(() => {
    changeToImage()
  }, [image])

  const detectionButton = <Button size="small" color="primary" onClick={changeToDetection}>
  Show Detection
</Button>

const showImageButton = <Button size="small" color="primary" onClick={changeToImage}>
  Show Image
</Button>

  function loadImage(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
  
      reader.onloadend = () => {
        resolve(reader.result);
      };
  
      reader.onerror = (error) => {
        reject(error);
      };
  
      reader.readAsDataURL(file);
    });
    }


  const handleImageChange = async (e) => {
    if (e.target.files && e.target.files[0]) {
      const img = e.target.files[0];
      setImage(img);

      const imgObject = new Image();

      // Set the source of the Image object to the URL of the image
      imgObject.src = URL.createObjectURL(img);

      // Set up the onload event to get the width and height once the image is loaded
      imgObject.onload = () => {
        setImageWidth(imgObject.width);
        setImageHeight(imgObject.height);
      };
      setBbs([])
      // changeToImage()
    }
  };

  async function getBlobFromPath(filePath) {
    try {
      // Fetch the file
      const response = await fetch(filePath);
  
      // Check if the request was successful (status code 200)
      if (!response.ok) {
        throw new Error('Failed to fetch file');
      }
  
      // Get the binary data as an ArrayBuffer
      const buffer = await response.arrayBuffer();
  
      // Create a Blob object from the ArrayBuffer
      const blob = new Blob([buffer]);
  
      return blob;
    } catch (error) {
      console.error('Error:', error.message);
      return null;
    }
  }

  function createFileFromBlob(blob, fileName) {
    try {
      // Create an array-like object with a single Blob
      const fileArray = [blob];
  
      // Create a File object from the array-like object
      const file = new File(fileArray, fileName, { type: "image/jpeg" });
  
      return file;
    } catch (error) {
      console.error('Error creating File:', error.message);
      return null;
    }
  }

  const handleImageChangeFromEx = async (imagePath) => {

    getBlobFromPath(imagePath)
    .then((blob) => {
      // const img = new File(blob, imagePath)
      const img = createFileFromBlob(blob, imagePath);
      setImage(img);

      const imgObject = new Image();

      // Set the source of the Image object to the URL of the image
      imgObject.src = imagePath


      // Set up the onload event to get the width and height once the image is loaded
      imgObject.onload = () => {
        setImageWidth(imgObject.width);
        setImageHeight(imgObject.height);
      };
      setBbs([])
    })
    

    
    // changeToImage()    
  };



  async function sendData(formData) {
    const response = await fetch('https://128.140.113.187:5000/api', {
      method: 'POST', 
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify(formData),
     });
    const data = await response.json();
    
    return data;
  }

  const uploadImage = async () => {
    if (!image) {
        alert("Please select an image first!");
        return;
    }

    const fileValue = await loadImage(image);
    const newFormData = {url : fileValue};
    setLoadingOrButton(CircularIndeterminate)
    const data = await sendData(newFormData);
    detectionImage = data.data_url
    listBB = data.text_bbs
    // const top = 100

    
    const c_width = 448
    const c_height = imageHeight*c_width/imageWidth
    const max_size = Math.max(c_width, c_height)

    setBbs([])
    for (let e_i = 0; e_i < listBB.length; e_i++) {
      const w = (listBB[e_i][0][3] - listBB[e_i][0][1])*c_width/imageWidth
      const h = (listBB[e_i][0][4] - listBB[e_i][0][2])*c_height/imageHeight
      const x1 = (listBB[e_i][0][1])*c_width/imageWidth
      const y1 = listBB[e_i][0][2]*c_height/imageHeight
      const text = listBB[e_i][1]
      setBbs(bbs => [...bbs, <Tooltip title={text} arrow><div style={{top: `${y1}px`,
                                            left: `${x1}px`,
                                            position: "absolute", 
                                            zIndex: 1000, 
                                            padding: "0px",
                                            border: "1px solid #00ff00", // Set the border properties (width, style, and color)
                                            borderRadius: "4px",
                                            minHeight: "0px",
                                            minWidth: "0px",
                                            height: `${h}px`,
                                            width: `${w}px`}}></div></Tooltip>])
    }

    setLoadingOrButton(detectionButton)

};



const inp = <Button component="label" variant="contained" startIcon={<CloudUploadIcon />}>
Upload file
<input type="file" ref={inputElement} onChange={handleImageChange} style={{ display: 'none' }}/>
</Button>

  return (
    <div>
      <StandardImageList setImage={handleImageChangeFromEx}></StandardImageList>
    
    <div style={{marginTop: '10px', marginBottom: '20px'}}>
      <MultiActionAreaCard image={previewUrl} buttonOne={uploadImage} loadingOrButton={loadingOrButton} inp={inp} bbs={bbs} ref={mediaElement}></MultiActionAreaCard>
    </div>
    </div>
  );
}

export default ImageUploader;