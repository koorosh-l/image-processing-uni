#/bin/sh

unzip "$1"

mv Test\ File/Test\ File/ test
mv Train\ File/Train\ File/ train
rm train/*.txt test/*.txt
rm Test\ File/ Train\ File/ -r
mkdir fruits
mv  test train fruits/

for i in ./fruits/train/*Apple*; do
    echo $i
    new_name=$(echo $i | sed "s/\ //")
    mv "$i" "$new_name"
done

for i in ./fruits/train/*Lichi*; do
    new_name=$(echo $i | sed "s/Lichi/Litchi/")
    mv "$i" "$new_name"
done


for i in ./fruits/train/*Pulm*; do
    new_name=$(echo $i | sed "s/\ Pulm/Plum/")
    mv "$i" "$new_name"
done
