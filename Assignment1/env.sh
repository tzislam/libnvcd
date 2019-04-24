export OMP_DISPLAY_ENV=false

preset=$1

sockets_close() {
    echo "exec policy: sockets_close"
    export OMP_PLACES=sockets
    export OMP_PROC_BIND=close
}

sockets_spread() {
    echo "exec policy: sockets_spread"
    export OMP_PLACES=sockets
    export OMP_PROC_BIND=spread
}

cores_close() {
    echo "exec policy: cores_close"
    export OMP_PLACES=cores
    export OMP_PROC_BIND=close
}

cores_spread() {
    echo "exec policy: cores_spread"
    export OMP_PLACES=cores
    export OMP_PROC_BIND=spread
}

case "$preset" in
    "")            sockets_close ;;
    sockets_close)  sockets_close ;;
    sockets_spread) sockets_spread ;;
    cores_close)    cores_close ;;
    cores_spread)   cores_spread ;;
    *)             echo "invalid argument" ;;
esac

