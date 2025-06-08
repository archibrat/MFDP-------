import logging
from typing import Dict, List
from fastapi import APIRouter, HTTPException, Depends
from sqlmodel import Session
from database.database import get_session
from models.mltask import MLTask, TaskStatus, MLTaskCreate
from services.rm.rm import rabbit_client
from services.rm.rpc import rpc_client
from services.logging.logging import get_logger
from services.crud.mltask import MLTaskService

logging.getLogger('pika').setLevel(logging.INFO)

logger = get_logger(logger_name=__name__)

ml_route = APIRouter()

def get_mltask_service(session: Session = Depends(get_session)) -> MLTaskService:
    return MLTaskService(session)

@ml_route.post(
    "/send_task", 
    response_model=Dict[str, str],
    summary="ML endpoint",
    description="Send ml request"
)
async def send_task(message: str, user_id: int, mltask_service: MLTaskService = Depends(get_mltask_service)) -> str:
    """
    Root endpoint returning welcome message.

    Returns:
        Dict[str, str]: Welcome message
    """
    created_task = None
    try:
        mltask = MLTaskCreate(question=message, user_id=user_id, status=TaskStatus.NEW)
        created_task = mltask_service.create(mltask)
        logger.info(f"Massage has created: {created_task}")

        logger.info(f"Sending task to RabbitMQ: {message}")
        rabbit_client.send_task(created_task)
        mltask_service.set_status(created_task.id, TaskStatus.QUEUED)
        return {"message": "Task sent successfully!"}
    except Exception as e:
        if created_task:
            mltask_service.set_status(created_task.id, TaskStatus.FAILED)
        logger.error(f"Unexpected error in sending task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@ml_route.post("/send_task_result", response_model=Dict[str, str])
def send_task_result(
    task_id: int,
    result: str,
    mltask_service: MLTaskService = Depends(get_mltask_service)
) -> Dict[str, str]:
    """
    Endpoint for sending ML task using Result.

    Args:
        message (str): The message to be sent.
        user_id (int): ID of the user creating the task.

    Returns:
        Dict[str, str]: Response message with original and processed text.
    """
    try:
        mltask_service.set_result(task_id, result)
        logger.info(f"!!!!!!!!Task result has been set: {result}")
        return {"message": "Task result sent successfully!"}
    except Exception as e:
        logger.error(f"Unexpected error in sending task result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    


@ml_route.post("/send_task_rpc", response_model=Dict[str, str])
async def send_task_rpc(
    message: str,
    user_id: int,
    mltask_service: MLTaskService = Depends(get_mltask_service)
) -> Dict[str, str]:
    """
    Endpoint for sending ML task using RPC.

    Args:
        message (str): The message to be sent.
        user_id (int): ID of the user creating the task.

    Returns:
        Dict[str, str]: Response message with original and processed text.
    """
    
    try:
        # Create task using service
        task_create = MLTaskCreate(
            question=message,
            user_id=user_id,
            status=TaskStatus.NEW
        )
        ml_task = mltask_service.create(task_create) 

        logger.info(f"Sending RPC request with message: {message}")
        result = rpc_client.call(text=message)
        logger.info(f"Received RPC response: {result}")

        # Update task with result using service
        mltask_service.set_result(ml_task.id, result)
        
        return {"original": message, "processed": result}
    except Exception as e:
        logger.error(f"Unexpected error in RPC call: {str(e)}")
        if ml_task:
            mltask_service.set_status(ml_task.id, TaskStatus.FAILED)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@ml_route.get("/tasks", response_model=List[MLTask])
async def get_all_tasks(
    mltask_service: MLTaskService = Depends(get_mltask_service)
):
    """Get all ML tasks."""
    return mltask_service.get_all()

@ml_route.get("/tasks/{task_id}", response_model=MLTask)
async def get_task(
    task_id: int,
    mltask_service: MLTaskService = Depends(get_mltask_service)
):
    """Get ML task by ID."""
    task = mltask_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task