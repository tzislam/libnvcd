export OMP_DISPLAY_ENV=true

preset=$1

socket_close() {
    echo "exec policy: socket_close"
    export OMP_PLACES=socket
    export OMP_PROC_BIND=close
}

socket_spread() {
    echo "exec policy: socket_spread"
    export OMP_PLACES=socket
    export OMP_PROC_BIND=spread
}

core_close() {
    echo "exec policy: close_close"
    export OMP_PLACES=core
    export OMP_PROC_BIND=close
}

core_spread() {
    echo "exec policy: core_spread"
    export OMP_PLACES=core
    export OMP_PROC_BIND=spread
}

case "$preset" in
    "")            socket_close ;;
    socket_close)  socket_close ;;
    socket_spread) socket_spread ;;
    core_close)    core_close ;;
    core_spread)   core_spread ;;
    *)             echo "invalid argument" ;;
esac

