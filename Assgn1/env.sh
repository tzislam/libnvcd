#!/bin/bash

export OMP_DISPLAY_ENV=true
export OMP_SCHEDULE='STATIC'

NUM_NODES=2
SOCKETS_PER_NODE=2
CORES_PER_SOCKET=6
THREADS_PER_CORE=2
CORES_PER_NODE=$(echo "${CORES_PER_SOCKET} * ${SOCKETS_PER_NODE}" | bc)
TOTAL_CORES=$(echo "${CORES_PER_NODE} * ${NUM_NODES}" | bc)
preset=$1

echo "TOTAL_CORES: " $TOTAL_CORES
echo "CORES PER NODE: " $CORES_PER_NODE

sockets_close() {
    echo "exec policy: sockets_close"
    export OMP_PLACES=sockets
    export OMP_PROC_BIND=close
    export OMP_NUM_THREADS=$CORES_PER_SOCKET
}

sockets_spread() {
    echo "exec policy: sockets_spread"
    export OMP_PLACES=sockets
    export OMP_PROC_BIND=spread
    export OMP_NUM_THREADS=$CORES_PER_NODE
}

cores_close() {
    echo "exec policy: cores_close"
    export OMP_PLACES=cores
    export OMP_PROC_BIND=close
    export OMP_NUM_THREADS=$THREADS_PER_CORE
}

cores_spread() {
    echo "exec policy: cores_spread"
    export OMP_PLACES=cores
    export OMP_PROC_BIND=spread
    export OMP_NUM_THREADS=$CORES_PER_SOCKET
}

case "$preset" in
    "")            sockets_close ;;
    sockets_close)  sockets_close ;;
    sockets_spread) sockets_spread ;;
    cores_close)    cores_close ;;
    cores_spread)   cores_spread ;;
    *)             echo "invalid argument" ;;
esac

