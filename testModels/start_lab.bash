# Instructions:
# make sure to create your own .venv to use the jupyter labs
#
# . jlab_venv/bin/activate
# $ jupyter lab

# to get started on pyserini on Linux machine
sudo apt remove -y openjdk-17-jdk
sudo apt autoremove
sudo apt update
sudo add-apt-repository -y ppa:linuxuprising/java
sudo apt update
sudo apt install -y openjdk-21-jdk
java --version

export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac))))
export PATH=$JAVA_HOME/bin:$PATH

echo "
from pyserini.index.lucene import LuceneIndexReader
" | python

jupyter lab
