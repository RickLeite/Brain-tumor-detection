docker build -t segment -f dockerfile .

docker run -v /volume:/app/data -it --name brain_segm segment bash
