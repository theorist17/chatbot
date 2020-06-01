cd chatbot

# conda
conda activate TL_ERC

# firewall
echo qwer1234 | sudo -S ufw disable
echo qwer1234 | sudo -S iptables -F

# ignore AVX warning
export TF_CPP_MIN_LOG_LEVEL=2 # https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
