import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';
import { Button, CardActionArea, CardActions } from '@mui/material';

export default function MultiActionAreaCard(props) {
  return (
    <Card sx={{maxWidth: 448, mx: 'auto'}}>
      
      <CardActionArea style={{zIndex: 1001}}>
      {props.bbs}
        <CardMedia
          component="img"
        //   height="140"
          image={props.image}
          alt=''
        >
        </CardMedia>
        

      </CardActionArea>
      <CardActions sx={{ marginTop: 'auto' }}>
        <Button size="small" color="primary" onClick={props.buttonOne}>
          Predict
        </Button>
        {props.loadingOrButton}
        
      </CardActions>
      <div sx={{ marginLeft: 'auto' }}>{props.inp}</div>
    </Card>
  );
}