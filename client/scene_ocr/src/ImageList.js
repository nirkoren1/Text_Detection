import React from 'react';
import ImageList from '@mui/material/ImageList';
import ImageListItem from '@mui/material/ImageListItem';
import { Button } from '@mui/material';

export default function StandardImageList(props) {
  return (
    <div style={{ display: 'flex', justifyContent: 'center', overflowX: 'auto', width: '100%' }}>
      <ImageList sx={{ width: '60%', height: 181 }} cols={itemData.length} rowHeight={164}>
        {itemData.map((item, index) => (
            // <button key={index} style={{padding: 0, border: 0}}>
            <img
              style={{maxHeight: 164, mx: 'auto', cursor: 'pointer'}}
              onClick={() => props.setImage(item.img)}
              srcSet={`${item.img}?w=164&h=164&fit=crop&auto=format&dpr=2 2x`}
              src={`${item.img}?w=164&h=164&fit=crop&auto=format`}
              alt={item.title}
              loading="lazy"
            />
            // </button>
        ))}
      </ImageList>
    </div>
  );
}

const itemData = [
  {
    img: 'grafity.jpg',
    title: 'grafity',
  },
  {
    img: 'large_andrae-ricketts-346066.jpg',
    title: 'Times Square',
  },
  {
    img: 'street.jpg',
    title: 'street sign',
  },
  {
    img: 'mall-shop.jpg',
    title: 'mall shop',
  },
  {
    img: 'london-buses.jpg',
    title: 'bus',
  },
  {
    img: 'newspapers.jpg',
    title: 'news papers',
  },
  {
    img: 'market2.jpg',
    title: 'market',
  },
  {
    img: 'cheese.jpg',
    title: 'cheese',
  },
  {
    img: 'books_covers.png',
    title: 'books_covers',
  },
  {
    img: 'document.jpg',
    title: 'document',
  },
  {
    img: 'somthing.jpg',
    title: 'somthing',
  },
  {
    img: 'code.jpg',
    title: 'code',
  },
];
