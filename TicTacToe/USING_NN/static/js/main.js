$(document).ready(function() {

    
    var canvas = $('canvas')[0]; // canvas
    var ctx = canvas.getContext("2d"); // context
    
    var width = canvas.width;
    var height = canvas.height;
    
    
    var board = [   0, 0, 0,
                    0, 0, 0,
                    0, 0, 0 ]; // board
   
    var human_btn = $("#human_btn");
    var ai_btn = $("#ai_btn");
    
    var human_art_isX = null;
    var isHumanTurn = false;
    var options = $(".options");


    var status = $("#status");

    
    var gameRunning = false;

    
    function hideOptions(doHide = true) {
        if(doHide) {
            options.hide();
        } else {
            options.show();
        }
    }


       function clearBoard() {
        ctx.clearRect(0, 0, width, height);
    }

       function drawBoard() {
        ctx.beginPath();
        ctx.moveTo(100,0);
        ctx.lineTo(100,300);
        ctx.moveTo(200,0);
        ctx.lineTo(200,300);
        ctx.moveTo(0,100);
        ctx.lineTo(300,100);
        ctx.moveTo(0,200);
        ctx.lineTo(300,200);
        ctx.stroke();
    }

       function resetBoard() {
        board = [   0, 0, 0,
                    0, 0, 0,
                    0, 0, 0 ];
    }

       function startGame() {
        resetBoard();
        hideOptions(true);
        drawBoard();
    }

    

       $(canvas).on('mousedown', function(evt) {
        if(!gameRunning || !isHumanTurn) return;
        isHumanTurn = false;
        var rect = canvas.getBoundingClientRect();
        var xpos = Math.floor((evt.clientX - rect.left) / (width/3)) ;
        var ypos = Math.floor((evt.clientY - rect.top) / (height/3));
        var completed = drawHumanAt(xpos,ypos);
               if(completed) {
            MachinePlay();
        } else {
            isHumanTurn = true
        }
    });


   
    function checkRoutineGameStatus() {

       
        var humanWon = false;
        var AIWon = false;
        isVacent = false;
        for(var i=0;i<board.length;i++) {
            if(board[i] == 0) {
                isVacent = true;
            }
        }

      
        for(var i=0;i<board.length;i+=3) {
            var rowsum = board[i] + board[i+1] + board[i+2];
            if(rowsum == 3) {
                AIWon = true;
            } else if(rowsum == -3) {
                humanWon = true;
            }
        }

       
        for(var i=0;i<3;i++) {
            var colsum = board[i] + board[i+3] + board[i+6];
            if(colsum == 3) {
                AIWon = true;
            } else if(colsum == -3) {
                humanWon = true;
            }
        }

      
        var sum_ld = board[0] + board[4] + board[8]
         if(sum_ld == 3) {
            AIWon = true;
        } else if(sum_ld == -3) {
            humanWon = true;
        }

        var sum_rd = board[2] + board[4] + board[6]
        if(sum_rd == 3) {
            AIWon = true;
        } else if(sum_rd == -3) {
            humanWon = true;
        }

        if(AIWon) {
            status.text("Ticky Won!!");
            gameRunning = false;
            hideOptions(false);
            return false;
        } else if(humanWon) {
            status.text("You Won!!!, Your Ingenious");
            gameRunning = false;
            hideOptions(false);
            return false;
        } else if(!isVacent) {
            status.text("Draw!!");
            gameRunning = false;
            hideOptions(false);
            return false;
        } else {
            return true;
        }

    } 

    
    var xdelta = 15 
    var ydelta = -15

   
    function drawHumanAt(xpos,ypos) {
        if(human_art_isX) {
            return drawAt('X',xpos,ypos,-1);
        } else {
            return drawAt('O',xpos,ypos,-1);
        }
    }

    
    function drawAt(character , xpos , ypos, player) {
        var index = xpos + ypos*3;
        if(board[index] == 0) {
            board[index] = player;
            ctx.font = "100px Arial";
            ctx.fillText(character, (xpos ) * (width / 3) + xdelta   , (ypos+1) * (height / 3   ) + ydelta );
        } else {
            return false;
        }
      
        return checkRoutineGameStatus();
    }

   

    
    function init() {
        clearBoard();
        drawBoard();
        hideOptions(false);
    }
    init();

   
    human_btn.on('click',function(e) {
        var human_art = $('input[name=art]:checked').val();
        if(human_art == 'X')
            human_art_isX = true;
        else
            human_art_isX = false;
        
        hideOptions();
        clearBoard();
        resetBoard();
        drawBoard();
        isHumanTurn = true;
        gameRunning = true;
        status.text("")

    });


       ai_btn.on('click',function() {
        var human_art = $('input[name=art]:checked').val();
        if(human_art == 'X')
            human_art_isX = true;
        else
            human_art_isX = false;
        
        hideOptions();
        resetBoard();
        clearBoard();
        drawBoard();
        isHumanTurn = false;
        gameRunning = true;
        status.text("")
        MachinePlay();

    })


   
    function MachinePlay() {
                $.ajax({
            type: 'POST',
            url: "/api/ticky",
            data: JSON.stringify({'data': board}),
            dataType: "json",
            contentType: "application/json",
            success: function(data) {
              
                xpos = data%3;
                ypos = Math.floor(data/3);
               
                if(human_art_isX) {
                     drawAt('O',xpos,ypos,1);
                } else {
                    drawAt('X',xpos,ypos,1);
                }
                isHumanTurn = true;
            }
        });    
    }

});
